#!/usr/bin/python3 -u

import tempfile
import os
import os.path as op
import sys
from multiprocessing import Pool
import argparse
import subprocess
import numpy as np
import pandas as pd
from utils_wgbs import IllegalArgumentError, eprint, segment_tool, add_GR_args, \
                       validate_file_list, validate_single_file, \
                       add_multi_thread_args, GenomeRefPaths, validate_local_exe, \
                       beta_sanity_check
from convert import add_bed_to_cpgs
from genomic_region import GenomicRegion, index2chrom
from beta_to_blocks import load_blocks_file
from numpy.typing import NDArray, ArrayLike


DEF_CHUNK = 60000


# TODO: remove it?
def is_block_file_nice(df):

    # no duplicated blocks
    if df.shape[0] != df.drop_duplicates().shape[0]:
        msg = 'Some blocks are duplicated'
        return False, msg

    # no overlaps between blocks
    sdf = df.sort_values(by='startCpG')
    if not (sdf['startCpG'][1:].values - sdf['endCpG'][:sdf.shape[0] - 1].values  >= 0).all():
        msg = 'Some blocks overlap'
        return False, msg

    return True, ''


def segment_process(params):
    sites = params['sites']
    start, end = sites
    assert end - start > 0, f'trying to segment an empty interval {sites}'
    if end - start == 1:
        return np.array([start, end])
    try:
        beta_files = ' '.join(params['betas'])
        cmd = f'{segment_tool} {beta_files} '
        cmd += f'-s {start - 1} -n {end - start} -max_cpg {params["max_cpg"]} '
        cmd += f' -ps {params["pcount"]} -max_bp {params["max_bp"]} '
        chrom = index2chrom(start, params["genome"])
        cmd = f'tabix {params["revdict"]} {chrom}:{start}-{end - 1} | cut -f2 |' + cmd
        lines = subprocess.check_output(cmd, shell=True).decode().strip().splitlines()

        segments = []
        for line in lines:
            try:
                s, e, score = line.strip().split()
                s, e = int(s) + start, int(e) + start  # adjust to genome coords
                score = float(score)
                segments.append((s, e, score))
            except ValueError:
                raise ValueError(f"Invalid segment line: {line}")

        # return np.array([s for s, _, _ in segments] + [segments[-1][1]])  # return breakpoints as before, sans scores
        return np.array(segments, dtype=np.float32)


    except Exception as e:
        eprint(f'Failed in sites {sites}')
        raise e


def simple_run(
    args,  # not ideal, but easier not to fix now
    betas
):
    # don't multiprocess
    # chunk, run the C++ iteratively
    # run segementation again between chunks to find overlap
    # stream output, save memory
    # match previous output format, with score on the end
    genome = GenomeRefPaths(args.genome)
    for beta in betas:
        if not beta_sanity_check(beta, genome):
            msg = f'[wt segment] ERROR: current genome reference ({genome.genome}) does not match the input beta file ({beta}).'
            raise IllegalArgumentError(msg)
    cpg_lower_bound = min(args.max_cpg, args.max_bp // 2)
    if cpg_lower_bound < 1:
        raise RuntimeError
    params = {'betas': betas,
              'pcount': args.pcount,
              'max_cpg': cpg_lower_bound,
              'max_bp': args.max_bp,
              'revdict': genome.revdict_path,
              'genome': genome}

    ### INPUT CHUNKING ###
    if args.chunk_size < args.max_cpg:
        msg = '[wt segment] WARNING: chunk_size is small compared to max_cpg and/or max_bp.\n' \
              '                      It may cause wt segment to fail. It\'s best setting\n' \
              '                      chunk_size > min{max_cpg, max_bp/2}'
        eprint(msg)

    if args.bed_file:
        bed_df = load_blocks_file(args.bed_file)[['startCpG', 'endCpG']].dropna()
        # make sure bed file has no overlaps or duplicated regions
        is_nice, is_nice_msg = is_block_file_nice(bed_df)
        if not is_nice:
            msg = '[wt segment] ERROR: invalid bed file.\n' \
                  f'                    {is_nice_msg}\n' \
                  f'                    Try: sort -k1,1 -k2,2n {args.bed_file} | ' \
                  'bedtools merge -i - | wgbstools convert --drop_empty -p -L -'
            eprint(msg)
            raise IllegalArgumentError('Invalid bed file')
        if bed_df.shape[0] > 2*1e4:
            msg = '[wt segment] WARNING: bed file contains many regions.\n' \
                  '                      Segmentation will take a long time.\n' \
                  '                      Consider running w/o -L flag and intersect the results\n'
            eprint(msg)
    else:   # No bed file provided
        gr = GenomicRegion(args)
        if gr.is_whole():  # more than one chrom
            cf = genome.get_chrom_cpg_size_table()  # chrom sizes
            if cf is None:
                raise RuntimeError
            cf['endCpG'] = np.cumsum(cf['size']) + 1  # add start col per chrom
            cf['startCpG'] = cf['endCpG'] - cf['size']  # add end col per chrom
            bed_df = cf[['startCpG', 'endCpG']]
        else:  # one region
            bed_df = pd.DataFrame(columns=['startCpG', 'endCpG'], data=[gr.sites])

    # having generated dataframe of regions above
    ### RUN BY REGION, BY CHUNK
    # don't merge between regions
    for _, row in bed_df.iterrows():  # row per region
        start, end = row
        chunk_borders = list(range(start, end, args.chunk_size)) + [end]
        chunk_segmentations: list[NDArray] = []
        for i in range(0, len(chunk_borders) - 1):
            s = chunk_borders[i]
            e = chunk_borders[i + 1]
            cp = params.copy()
            cp['sites'] = (s, e)  # overwrite sites to create chunk specific params
            chunk_segmentations.append(segment_process(cp))
        # breakpoint()

        # run 2 chunks, then run over boundary of those chunks to merge
        # merge chunk results
        # in neighbour pairs to prioritise preservation of local information
        # (for loop wrapped in while loop)
        merging_segmentations: list[NDArray] = chunk_segmentations
        while len(merging_segmentations) != 1:
            pairwise_merges: list[NDArray] = []
            for i in range(0, len(merging_segmentations) - 1, 2):
                # merge candidate chunks
                mc1 = merging_segmentations[i]
                mc2 = merging_segmentations[i + 1]
                if mc1[-1][1] != mc2[0][0]:
                    # breakpoint()
                    msg = '[wt segment] Chunk stitching Failed! ' \
                          '             chunks are not adjacent'
                    raise IllegalArgumentError(msg)

                n1 = mc1[-1][1] - mc1[0][0]  # size, end of chunk - start
                n2 = mc2[-1][1] - mc2[0][0]  # as above
                patch1_size = min(50, n1)
                patch2_size = min(50, n2)
                patch = np.array([], dtype=int)
                while patch1_size <= n1 and patch2_size <= n2:
                    # calculate blocks for patch:
                    start = mc1[-1][1] - patch1_size #- 1
                    end = mc1[-1][1] + patch2_size
                    patch_params = dict(params, **{'sites': (start, end)})
                    patch = segment_process(patch_params)

                    # find the overlaps
                    o1: int | None = find_overlap(mc1, patch)
                    o2 = find_overlap(mc2, patch)
                    if o1 is not None and o2 is not None:
                        # successful stitch with patches
                        # as in the original implementation, scoring is not considered when merging
                        merged = merge_segmentations(mc1, patch, o1)
                        merged = merge_segmentations(merged, mc2, o2)
                        pairwise_merges.append(merged)
                        break
                    else:
                        # failed stitch - increase patch sizes
                        if o1 is None:
                            patch1_size = increase_patch(patch1_size, n1)
                        if o2 is None:
                            patch2_size = increase_patch(patch2_size, n2)
                else:
                    # Failed: could not stich the two chuncks
                    msg = '[wt segment] Patch stitching Failed! ' \
                          '             Try increasing chunk size (--chunk_size flag)'
                    raise IllegalArgumentError(msg)
                # breakpoint()
            odd_straggler = [merging_segmentations[-1]] if len(merging_segmentations) % 2 else []
            merging_segmentations = pairwise_merges + odd_straggler
            # breakpoint()
        # breakpoint()
        ### TODO: WRITE PER REGION
    # breakpoint()

def find_overlap(df1: NDArray, df2: NDArray) -> int | None:
    # get starts and ends as 1D array for each df
    # make that 1D array unique
    # then check for overlap between the two
    df1_bps = np.unique(df1[1:, :2].flatten())  # breakpoints
    df2_bps = np.unique(df2[:-1, :2].flatten())
    dups = np.isin(df1_bps, df2_bps)
    if np.any(dups):
        return df1_bps[dups][0]
    else:
        return None

def merge_segmentations(df1: NDArray, df2: NDArray, overlap: int) -> NDArray:
    df1_ol_idx = np.argwhere(df1[:, 1] == overlap)  # first occurence should be an end, since each coord should appear twice, once as an end, once as a start
    df2_ol_idx = np.argwhere(df2[:, 0] == overlap)

    if df1_ol_idx.size != 1 or df2_ol_idx.size != 1:
        raise ValueError
    else:
        df1_row = df1_ol_idx[0][0]  # where overlap is segement end
        df2_row = df2_ol_idx[0][0]  # where start

    return np.concatenate([df1[:df1_row + 1], df2[df2_row:]])


class SegmentByChunks:
    def __init__(self, args, betas):
        self.betas = betas
        max_cpg = min(args.max_cpg, args.max_bp // 2)
        assert max_cpg > 1
        self.genome = GenomeRefPaths(args.genome)
        self.param_dict = {'betas': betas,
                          'pcount': args.pcount,
                          'max_cpg': max_cpg,
                          'max_bp': args.max_bp,
                          'revdict': self.genome.revdict_path,
                          'genome': self.genome
                          }
        self.args = args
        self.validate_genome()

    def validate_genome(self):
        for beta in self.betas:
            if not beta_sanity_check(beta, self.genome):
                msg = f'[wt segment] ERROR: current genome reference ({self.genome.genome}) does not match the input beta file ({beta}).'
                raise IllegalArgumentError(msg)

    def break_to_chunks(self):
        """ Break range of sites to chunks of size 'step',
            while keeping chromosomes separated """
        # print a warning in case chunk size is too small
        step = self.args.chunk_size
        if step < self.args.max_cpg:
            msg = '[wt segment] WARNING: chunk_size is small compared to max_cpg and/or max_bp.\n' \
                  '                      It may cause wt segment to fail. It\'s best setting\n' \
                  '                      chunk_size > min{max_cpg, max_bp/2}'
            eprint(msg)

        if self.args.bed_file:
            df = load_blocks_file(self.args.bed_file)[['startCpG', 'endCpG']].dropna()
            # make sure bed file has no overlaps or duplicated regions
            is_nice, msg = is_block_file_nice(df)
            if not is_nice:
                msg = '[wt segment] ERROR: invalid bed file.\n' \
                      f'                    {msg}\n' \
                      f'                    Try: sort -k1,1 -k2,2n {self.args.bed_file} | ' \
                      'bedtools merge -i - | wgbstools convert --drop_empty -p -L -'
                eprint(msg)
                raise IllegalArgumentError('Invalid bed file')
            if df.shape[0] > 2*1e4:
                msg = '[wt segment] WARNING: bed file contains many regions.\n' \
                      '                      Segmentation will take a long time.\n' \
                      '                      Consider running w/o -L flag and intersect the results\n'
                eprint(msg)

        else:   # No bed file provided
            gr = GenomicRegion(self.args)
            # whole genome - make a dummy "bed file" of the full chromosomes
            if gr.is_whole():
                cf = self.genome.get_chrom_cpg_size_table()
                cf['endCpG'] = np.cumsum(cf['size']) + 1
                cf['startCpG'] = cf['endCpG'] - cf['size']
                df = cf[['startCpG', 'endCpG']]
            # one region
            else:
                df = pd.DataFrame(columns=['startCpG', 'endCpG'], data=[gr.sites])

        # build a DataFrame of chunks, with a "tag"/label field,
        # so we know which chunks to merge later on.
        tags = []
        starts = []
        ends = []
        for _, row in df.iterrows():
            start, end = row
            bords = list(range(start, end, step)) + [end]
            tags += [f'{start}-{end}'] * (len(bords) -1)
            starts += bords[:-1]
            ends += bords[1:]
        return tags, starts, ends

    def run(self):
        # break input region/s to small chunks
        tags, starts, ends = self.break_to_chunks()
        # segment each chunk separately in a single thread
        p = Pool(self.args.threads)
        params = [(dict(self.param_dict, **{'sites': (s, e)}),) for s, e in zip(starts, ends)]
        arr = p.starmap(segment_process, params)
        p.close()
        p.join()
        breakpoint()

        # merge chunks from the same "tag" group
        # (i.e. the same chromosome, or the same region of the provided bed file)
        df = pd.DataFrame()
        for tag in set(tags):
            carr = [arr[i] for i in range(len(arr)) if tags[i] == tag]
            breakpoint()
            merged = self.merge_df_list(carr)
            df = pd.concat([df, pd.DataFrame({'startCpG': merged[:-1], 'endCpG': merged[1:]})])
        breakpoint()
        self.dump_result(df.reset_index(drop=True))

    def merge_df_list(self, dflist):
        # Given a set of chunks to merge, recursively pairwise stich them.

        while len(dflist) > 1:
            p = Pool(self.args.threads)
            params = [(dflist[i - 1], dflist[i], self.param_dict) for i in range(1, len(dflist), 2)]
            # added single initial call outside of pool for debug/testing
            stitch_2_dfs(params[0][0], params[0][1], self.param_dict)
            breakpoint()
            arr = p.starmap(stitch_2_dfs, params)
            p.close()
            p.join()

            last_df = [dflist[-1]] if len(dflist) % 2 else []
            dflist = arr + last_df
        return dflist[0]

    def dump_result(self, df):
        if df.empty:
            eprint('Empty blocks array')
            return

        # sort by startCpG and filter by CpGs
        nr_blocks = df.shape[0]
        df.sort_values(by=['startCpG'], inplace=True)
        df = df[df.endCpG - df.startCpG > self.args.min_cpg - 1].reset_index(drop=True)

        # verbose
        nr_blocks_filt = df.shape[0]
        nr_dropped = nr_blocks - nr_blocks_filt
        eprint(f'[wt segment] found {nr_blocks_filt:,} blocks\n' \
               f'             (dropped {nr_dropped:,} short blocks)')

        # add genomic loci and dump/print
        temp_path = next(tempfile._get_candidate_names())
        try:
            df.to_csv(temp_path, sep='\t', header=None, index=None)
            add_bed_to_cpgs(temp_path, self.genome.genome, self.args.out_path)
        finally:
            if op.isfile(temp_path):
                os.remove(temp_path)


#############################################################
#                                                           #
#           Chunk stiching logic                            #
#                                                           #
#############################################################

def stitch_2_dfs(b1, b2, params):

    # if b2 is not the direct extension of b1, we have a problem
    breakpoint()
    if b1[-1] != b2[0]:
        msg = '[wt segment] Patch stitching Failed! ' \
              '             patches are not supposed to be merged'
        raise IllegalArgumentError(msg)

    n1 = b1[-1] - b1[0]
    n2 = b2[-1] - b2[0]
    patch1_size = min(50, n1)
    patch2_size = min(50, n2)
    patch = np.array([], dtype=int)
    while patch1_size <= n1 and patch2_size <= n2:
        # calculate blocks for patch:
        start = b1[-1] - patch1_size #- 1
        end = b1[-1] + patch2_size
        cparams = dict(params, **{'sites': (start, end)})
        patch = segment_process(cparams)

        # find the overlaps
        if is_2_overlap(b1, patch) and is_2_overlap(patch, b2):
            # successful stitch with patches
            return merge2(merge2(b1, patch), b2)
        else:
            # failed stitch - increase patch sizes
            if not is_2_overlap(b1, patch):
                patch1_size = increase_patch(patch1_size, n1)
            if not is_2_overlap(patch, b2):
                patch2_size = increase_patch(patch2_size, n2)

    # Failed: could not stich the two chuncks
    msg = '[wt segment] Patch stitching Failed! ' \
          '             Try increasing chunk size (--chunk_size flag)'
    raise IllegalArgumentError(msg)


def stitch_scored_chunks(b1: NDArray, b2: NDArray, params: dict):
    # if b2 is not the direct extension of b1, we have a problem
    breakpoint()
    if b1[-1][1] != b2[0][0]:
        msg = '[wt segment] Patch stitching Failed! ' \
              '             patches are not supposed to be merged'
        raise IllegalArgumentError(msg)

    n1 = b1[-1][1] - b1[0][0]  # size, end of chunk - start
    n2 = b2[-1][1] - b2[0][0]  # as above
    patch1_size = min(50, n1)
    patch2_size = min(50, n2)
    patch = np.array([], dtype=int)
    while patch1_size <= n1 and patch2_size <= n2:
        # calculate blocks for patch:
        start = b1[-1][1] - patch1_size #- 1
        end = b1[-1][1] + patch2_size
        cparams = dict(params, **{'sites': (start, end)})
        patch = segment_process(cparams)

        # find the overlaps
        if is_2_overlap(b1, patch) and is_2_overlap(patch, b2):
            # successful stitch with patches
            return merge2(merge2(b1, patch), b2)
        else:
            # failed stitch - increase patch sizes
            if not is_2_overlap(b1, patch):
                patch1_size = increase_patch(patch1_size, n1)
            if not is_2_overlap(patch, b2):
                patch2_size = increase_patch(patch2_size, n2)

    # Failed: could not stich the two chuncks
    msg = '[wt segment] Patch stitching Failed! ' \
          '             Try increasing chunk size (--chunk_size flag)'
    raise IllegalArgumentError(msg)


def is_2_overlap(b1, b2):
    return np.sum(find_dups(b1, b2))


def find_dups(b1, b2):
    return pd.Series(np.concatenate([b1, b2])).duplicated(keep=False).values


def merge2(b1, b2):
    nr_from_df1 = np.argmax(find_dups(b1, b2))
    skip_from_df2 = np.searchsorted(b2, b1[nr_from_df1])
    return np.concatenate([b1[:nr_from_df1 + 1], b2[skip_from_df2 + 1:]]).copy()


def increase_patch(pre_size, maxval):
    if pre_size == maxval:
        return maxval + 1  # too large, so the while loop will break
    return int(min(pre_size * 2, maxval))


#############################################################
#                                                           #
#                       Main                                #
#                                                           #
#############################################################

def parse_args():
    parser = argparse.ArgumentParser(description=main.__doc__)
    add_GR_args(parser, bed_file=True)
    betas_or_file = parser.add_mutually_exclusive_group(required=True)
    betas_or_file.add_argument('--betas', nargs='+')
    betas_or_file.add_argument('--beta_file', '-F')
    parser.add_argument('-c', '--chunk_size', type=int, default=DEF_CHUNK,
                        help=f'Chunk size. Default {DEF_CHUNK} sites')
    parser.add_argument('-p', '--pcount', type=float, default=15,
                        help='Pseudo counts of C\'s and T\'s in each block. Default 15')
    parser.add_argument('--min_cpg', type=int, default=1,
                        help='Minimal block size (in #sites) to output. Shorter blocks will simply be ' \
                             'ommited from output (equivalent to set min_cpg to 1 and then filter output by ' \
                             'length). Default is 1')
    parser.add_argument('--max_cpg', type=int, default=1000,
                        help='Maximal allowed block size (in #sites). Default is 1000')
    parser.add_argument('--max_bp', type=int, default=2000,
                        help='Maximal allowed block size (in bp). Default is 2000')
    parser.add_argument('-o', '--out_path', default=sys.stdout,
                        help='output path [stdout]')
    add_multi_thread_args(parser)
    return parser.parse_args()


def parse_betas_input(args):
    """
    parse user input to get the list of beta files to segment
    Either args.betas is a list of beta files,
    or args.beta_file is a text file in which each line is a beta file
    return: list of beta files
    """
    if args.betas:
        betas = args.betas
    elif args.beta_file:
        validate_single_file(args.beta_file)
        with open(args.beta_file, 'r') as f:
            betas = [b.strip() for b in f.readlines() if b.strip() and not b.startswith('#')]
        if not betas:
            raise IllegalArgumentError(f'no beta files found in file {args.beta_file}')
    validate_file_list(betas)
    return betas


def main():
    """
    Segment the genome, or a subset region, to homogenously methylated blocks.
    Input: one or more beta files to segment
    Output: blocks file (BED format + startCpG, endCpG columns)
    """
    args = parse_args()
    validate_local_exe(segment_tool)
    betas = parse_betas_input(args)
    simple_run(
        args,
        betas
    )
    # SegmentByChunks(args, betas).run()


if __name__ == '__main__':
    main()
