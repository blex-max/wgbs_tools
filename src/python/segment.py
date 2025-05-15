#!/usr/bin/python3 -u

import tempfile
import os
import sys
import csv
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
from numpy.typing import NDArray
import math


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
    gn_start, gn_end = sites
    if not isinstance(gn_start, int) or not isinstance(gn_end, int):
        raise TypeError(f'start and end should be type int, not {type(gn_start)}, {type(gn_end)}')

    assert gn_end - gn_start > 0, f'trying to segment an empty interval {sites}'

    if gn_end - gn_start == 1:
        return np.array([gn_start, gn_end])

    beta_files = ' '.join(params['betas'])
    cmd = f'{segment_tool} {beta_files} '
    cmd += f'-s {gn_start - 1} -n {gn_end - gn_start} -max_cpg {params["max_cpg"]} '  # testing no -1
    cmd += f' -ps {params["pcount"]} -max_bp {params["max_bp"]} '
    chrom = index2chrom(gn_start, params["genome"])
    cmd = f'tabix {params["revdict"]} {chrom}:{gn_start}-{gn_end - 1} | cut -f2 |' + cmd

    try:
        proc = subprocess.run(
                            cmd,
                            shell=True,
                            capture_output=True,
                            text=True,
                            check=True
                        )
    except subprocess.CalledProcessError as e:
        eprint(f'Call to segmentor failed over {sites}')
        print(f'Dumping stderr:\n{e.stderr}')
        print(f'Dumping stdout:\n{e.stdout}')
        raise e
    except Exception as e:
        eprint(f'Call to segmentor failed over {sites}')
        raise e

    lines = proc.stdout.strip().splitlines()
    segments = []
    for line in lines:
        try:
            block_start_str, block_end_str, score_str = line.strip().split()
            gn_block_start = int(block_start_str) + gn_start
            gn_block_end = int(block_end_str) + gn_start  # ab: adjust to genome coords
            score = float(score_str)
            if gn_block_start < gn_start:
                RuntimeError('block underflow!')
            segments.append((gn_block_start, gn_block_end, score))
        except ValueError:
            raise ValueError(f"Invalid segment line: {line}")

    # return np.array([s for s, _, _ in segments] + [segments[-1][1]])  # ab: return breakpoints as before, sans scores
    # ab: ensuring 64 bit precision is critical to avoid bugs
    return np.array(segments, dtype=np.float64)


def run(
    args,  # ab: not ideal, but easier not to fix now
    betas
):
    # ab: establish tmp early, so we can fail early
    if not os.path.isdir(args.work_dir):
        raise RuntimeError('work dir does not appear to be a directory')
    if not os.access(args.work_dir, os.R_OK | os.W_OK):
        raise RuntimeError("Don't appear to have appropriate permissions on work dir")
    # ab: no mp, chunk, run the C++ iteratively
    # ab: match previous output format, with score on the end
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

    ### ab: INPUT CHUNKING ###
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
        if gr.is_whole():  # ab: more than one chrom
            cf = genome.get_chrom_cpg_size_table()  # ab: chrom sizes
            if cf is None:
                raise RuntimeError
            cf['endCpG'] = np.cumsum(cf['size']) + 1  # ab: add start col per chrom
            cf['startCpG'] = cf['endCpG'] - cf['size']  # ab: add end col per chrom
            bed_df = cf[['startCpG', 'endCpG']]
        else:  # one region
            bed_df = pd.DataFrame(columns=pd.Index(['startCpG', 'endCpG']), data=[gr.sites])

    # ab:
    # generated dataframe of regions above
    ### RUN BY REGION, BY CHUNK
    # don't merge between regions
    result_df = pd.DataFrame(columns=pd.Index(['startCpG', 'endCpG', 'score']))
    for _, row in bed_df.iterrows():  # ab: row per region
        start, end = row
        chunk_borders = list(range(start, end, args.chunk_size)) + [end]
        chunk_segmentations: list[NDArray] = []
        for i in range(0, len(chunk_borders) - 1):
            chunk_s = chunk_borders[i]
            chunk_e = chunk_borders[i + 1]
            cp = params.copy()
            cp['sites'] = (chunk_s, chunk_e)  # ab: overwrite sites to create chunk specific params
            chunk_segmentations.append(segment_process(cp))


        # ab:
        # merge chunks; run segementation again between chunks to find overlap
        # merge neighbour pairs to prioritise preservation of local information
        merging_segmentations: list[NDArray] = chunk_segmentations
        maxiter = math.ceil(math.log2(len(merging_segmentations))) + 1
        niter = 0
        while len(merging_segmentations) != 1:
            if niter > maxiter:
                raise RuntimeError('chunk merging seems to have exceeded a sensible number of iterations')  # ab: panic
            pairwise_merges: list[NDArray] = []
            for i in range(0, len(merging_segmentations) - 1, 2):
                # ab: mc = "merge candidate" chunks
                mc1: NDArray = merging_segmentations[i]
                mc2 = merging_segmentations[i + 1]
                if mc1[-1][1] != mc2[0][0]:
                    msg = '[wt segment] Chunk stitching Failed! ' \
                          '             chunks are not adjacent'
                    raise IllegalArgumentError(msg)

                n1 = int(mc1[-1][1] - mc1[0][0])  # ab: size, end of chunk - start
                n2 = int(mc2[-1][1] - mc2[0][0])  # ab: as above
                patch1_size = min(50, n1)
                patch2_size = min(50, n2)
                while patch1_size < n1 and patch2_size < n2:
                    # calculate blocks for patch:
                    start = int(mc1[-1][1] - patch1_size) #- 1
                    end = int(mc1[-1][1] + patch2_size)
                    patch_params = dict(params, **{'sites': (start, end)})
                    patch: NDArray = segment_process(patch_params)

                    # find the overlaps
                    o1: int | None = find_overlap(mc1, patch)
                    o2 = find_overlap(mc2, patch)
                    if o1 is not None and o2 is not None:
                        # successful stitch with patches
                        # as in the original implementation, scoring is not considered when merging
                        merged = overlap_merge(mc1, patch, o1)
                        merged = overlap_merge(merged, mc2, o2)
                        pairwise_merges.append(merged)
                        break
                    else:
                        # failed stitch - increase patch sizes
                        if o1 is None:
                            patch1_size = increase_patch(patch1_size, n1)
                        if o2 is None:
                            patch2_size = increase_patch(patch2_size, n2)
                else:
                    msg = '[wt segment] Patch stitching Failed! ' \
                          '             Try increasing chunk size (--chunk_size flag)'
                    raise IllegalArgumentError(msg)
            odd_straggler = [merging_segmentations[-1]] if len(merging_segmentations) % 2 else []
            merging_segmentations = pairwise_merges + odd_straggler
            if len(merging_segmentations) < 1:
                raise RuntimeError('chunk merging failed for unknown reason')  # ab: panic
            niter += 1
        if result_df.empty:  # ab: satisfy pandas futurewarning about upcoming changes to concat
            result_df = pd.DataFrame(merging_segmentations[0], columns=result_df.columns)
        else:
            result_df = pd.concat([result_df, pd.DataFrame(merging_segmentations[0], columns=result_df.columns)], ignore_index=True)
    # ab: convert result and dump; lifted from original implemenation insofar as was possible; to avoid introducing discrepancies
    nr_blocks = result_df.shape[0]
    result_df.sort_values(by=['startCpG'], inplace=True)
    result_df = result_df[result_df.endCpG - result_df.startCpG >= args.min_cpg].reset_index(drop=True)

    nr_blocks_filt = result_df.shape[0]
    nr_dropped = nr_blocks - nr_blocks_filt
    eprint(f'[wt segment] found {nr_blocks_filt:,} blocks\n' \
           f'             (dropped {nr_dropped:,} short blocks)')

    # ab: add genomic loci - at present this requires dumping to disk
    result_unscored = result_df.iloc[:, :2]  # ab: drop scores
    # ab: n.b. where it's necessary to use tempobj.name rather than tempobj directly
    try:
        loci_td = tempfile.TemporaryDirectory(dir=args.work_dir, prefix="segtmp", ignore_cleanup_errors=True)
        unscored_tmp = tempfile.NamedTemporaryFile(dir=loci_td.name, prefix="tmp_", suffix=".segdat")
        unscored_gn_tmp = tempfile.NamedTemporaryFile(dir=loci_td.name, prefix="tmp_", suffix=".segdat")  # ab: with genomic loci
        result_unscored.to_csv(unscored_tmp, sep='\t', header=None, index=None)
        add_bed_to_cpgs(unscored_tmp.name, genome.genome, unscored_gn_tmp.name)
        result_w_loci = pd.read_csv(unscored_gn_tmp, sep='\t', header=None, dtype=object)
    except Exception as e:
        raise RuntimeError(f'Failed to add genomic coordinates to segments, reporting {e}')
    finally:
        if 'unscored_tmp' in locals(): unscored_tmp.close(); del unscored_tmp
        if 'unscored_gn_tmp' in locals(): unscored_gn_tmp.close(); del unscored_gn_tmp
        if 'td' in locals(): loci_td.cleanup(); del loci_td

    if result_w_loci.shape[0] != result_df.shape[0]:
        raise RuntimeError('number of blocks unexpectedly changed after adding genomic loci')
    final_df = pd.concat([result_w_loci, result_df.iloc[:, -1]], axis=1)  # ab: re-add scoring
    final_df.to_csv(str(args.out_path), sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE)


# ab: new merge funcs to handle scored data
def find_overlap(arr1: NDArray, arr2: NDArray) -> int | None:
    """
    check for identical value in segment ends of df1 and segment starts of df2
    """
    arr1_ends = np.unique(arr1[1:, 1].flatten())
    arr2_starts = np.unique(arr2[:-1, 0].flatten())
    dups = np.isin(arr1_ends, arr2_starts)
    if np.any(dups):
        return int(arr1_ends[dups][0])
    else:
        return None


# ab: new merge funcs
# ab:
# extremely rare edge case where this merge causes the output not to match original implementation
# 1 occurence over all blocks reported for hg19
# I have not looked into the cause at this time
# I won't do so unless it's requested/it's decided the discrepancy is non-trivial
# It's almost certainly to do with the assumption that the breakpoint should first be an end in arr1
# then a start in arr2, or because we're keeping the overlap row from arr1
# and in the original implementation it keeps from arr2, or some combination of these possibilites
def overlap_merge(arr1: NDArray, arr2: NDArray, overlap: int) -> NDArray:
    """
    given unique segment coordinate (overlap) where two dataframes should overlap,
    find rows where overlap coordinate appears in df1 segment end coordinates
    and df2 segment start coordinates, and merge at that point
    """
    arr1_ol_i = np.argwhere(arr1[:, 1] == overlap)  # ab: ends
    arr2_ol_i = np.argwhere(arr2[:, 0] == overlap)  # ab: start

    if arr1_ol_i.size < 1 or arr2_ol_i.size < 1:
        raise RuntimeError('overlap not found in chunks to merge!')
    if arr1_ol_i.size != 1 or arr2_ol_i.size != 1:
        raise RuntimeError('overlap coordinate appears more than once in chunk')

    arr1_row = int(arr1_ol_i[0][0])  # ab: where overlap is segement end
    arr2_row = int(arr2_ol_i[0][0])  # ab: where start

    arr1_row += 1 # retain overlap row from df1

    return np.concatenate([arr1[:arr1_row], arr2[arr2_row:]])


def increase_patch(pre_size, maxval):
    if pre_size == maxval:
        return maxval + 1  # too large, so the while loop will break
    return int(min(pre_size * 2, maxval))


#############################################################
#                                                           #
#                       Main                                #
#                                                           #
#############################################################


def main():
    """
    Segment the genome, or a subset region, to homogenously methylated blocks.
    Input: one or more beta files to segment
    Output: blocks file (BED format + startCpG, endCpG columns)
    """
    validate_local_exe(segment_tool)

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
    parser.add_argument("--work_dir", default=os.getcwd(),
                        help="directory in which to create temp directory, for storge temporary working files. Defaults to cwd")
    add_multi_thread_args(parser)
    args = parser.parse_args()

    # Either args.betas is a list of beta files,
    # or args.beta_file is a text file in which each line is a beta file
    if args.betas:
        betas = args.betas
    elif args.beta_file:
        validate_single_file(args.beta_file)
        with open(args.beta_file, 'r') as f:
            betas = [b.strip() for b in f.readlines() if b.strip() and not b.startswith('#')]
        if not betas:
            raise IllegalArgumentError(f'no beta files found in file {args.beta_file}')
    validate_file_list(betas)

    run(
        args,
        betas
    )


if __name__ == '__main__':
    main()
