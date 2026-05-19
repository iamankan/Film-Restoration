import numpy as np
from typing import Dict, List
from scipy.spatial import KDTree

def make_kd_tree(slice_map:Dict):
    _metadata = []
    _points_list = []
    for _seg_no, _points in slice_map.items():
        for _point_idx, _point in enumerate(_points):
            _metadata.append({
                'seg_no': _seg_no,
                'point': {
                    'idx': _point_idx,
                    'coordinate': _point
                    }
            })
            _points_list.append(_point)
    _kd_tree = KDTree(data=np.array(_points_list))
    return _kd_tree, _metadata


def get_dummy_vol_map():
    vol_map = {
        # Slice 0: Two parallel segments starting out
        0: {
            0: np.array([[10, 20, 0], [11, 20, 0], [12, 20, 0]]),
            1: np.array([[15, 20, 0], [16, 20, 0], [17, 20, 0]])
        },
        # Slice 1: Continuing smoothly (Z incremented)
        1: {
            0: np.array([[10, 20, 1], [11, 20, 1], [12, 20, 1]]),
            1: np.array([[15, 20, 1], [16, 20, 1], [17, 20, 1]])
        },
        # Slice 2: MERGE EVENT BEGINS - They drift closer together
        2: {
            0: np.array([[11, 20, 2], [12, 20, 2], [13, 20, 2]]),
            1: np.array([[14, 20, 2], [15, 20, 2], [16, 20, 2]])
        },
        # Slice 3: FULLY MERGED - Fused into a single continuous Segment 0
        3: {
            0: np.array([[11, 20, 3], [12, 20, 3], [13, 20, 3], [14, 20, 3], [15, 20, 3], [16, 20, 3]])
        },
        # Slice 4: Staying merged, drifting slightly in the X-axis
        4: {
            0: np.array([[12, 20, 4], [13, 20, 4], [14, 20, 4], [15, 20, 4], [16, 20, 4], [17, 20, 4]])
        },
        # Slice 5: SPLIT EVENT BEGINS - Gaps start forming in the center coordinates
        5: {
            0: np.array([[12, 20, 5], [13, 20, 5]]),
            1: np.array([[16, 20, 5], [17, 20, 5]])
        },
        # Slice 6: Fully split into two separate paths, plus a rogue Layer 3 appears
        6: {
            0: np.array([[11, 20, 6], [12, 20, 6]]),
            1: np.array([[17, 20, 6], [18, 20, 6]]),
            2: np.array([[25, 30, 6], [26, 30, 6]]) # Stray fragment (Segment 2)
        },
        # Slice 7: Tracking the split layers and the stray fragment
        7: {
            0: np.array([[11, 20, 7], [12, 20, 7]]),
            1: np.array([[17, 20, 7], [18, 20, 7]]),
            2: np.array([[25, 30, 7], [26, 30, 7]])
        },
        # Slice 8: TERMINATION EVENT - Segment 2 breaks off completely (0 points)
        8: {
            0: np.array([[10, 20, 8], [11, 20, 8]]),
            1: np.array([[18, 20, 8], [19, 20, 8]])
            # Segment 2 is missing here!
        },
        # Slice 9: Final slice tracking just the two surviving main branches
        9: {
            0: np.array([[10, 20, 9], [11, 20, 9]]),
            1: np.array([[18, 20, 9], [19, 20, 9]])
        }
    }

    return vol_map

def get_match_tree(matches, curr_metadata, next_metadata, curr_slc_idx):
    print(f'The mappings:\n')
    for _i, _mat in enumerate(matches):
        for _matp in _mat:
            print(f'[Slice: {curr_slc_idx}, Segment: {curr_metadata[_i]['seg_no']}, Point: {curr_metadata[_i]['point']['idx']}] -> [Slice: {curr_slc_idx+1}, Segment: {next_metadata[_matp]['seg_no']}, Point: {next_metadata[_matp]['point']['idx']}]\n')





def main():
    # vol_map = {
    #     0:{
    #         0:np.array(
    #             [
    #                 [0,0,0], [1,0,0], [2,0,0]
    #             ]
    #         ),
    #         1:np.array(
    #             [
    #                 [3,0,0], [4,0,0], [5,0,0]
    #             ]
    #         )
    #     },
    #     1:{
    #         0:np.array(
    #             [
    #                 [1,0,1], [2,0,1], [3,0,1], [4,0,1], [5,0,1]
    #             ]
    #         )
    #     }
    # }

    vol_map = get_dummy_vol_map()

    _kd_map = {}

    for _slice_idx,_slice_map in vol_map.items():
        _kd_tree, _kd_meta = make_kd_tree(slice_map=_slice_map)
        _kd_map[_slice_idx] = {
            'metadata': _kd_meta,
            'tree': _kd_tree
        }
    
    total_slices = len(_kd_map.keys())

    _match_map = []
    
    for _curr_slc_idx, _curr_slc_map in _kd_map.items():
        if _curr_slc_idx == total_slices-1:
            break
        print(f'Curr slice index: {_curr_slc_idx}/{total_slices-1}')
        _curr_slc = _curr_slc_map
        _next_slc = _kd_map[_curr_slc_idx + 1]
        _matches = _curr_slc['tree'].query_ball_tree(other=_next_slc['tree'], r=1)
        _match_map.append(_matches)
        print(f'Matches: {_matches}')
        get_match_tree(matches=_matches, curr_metadata=_curr_slc['metadata'], next_metadata=_next_slc['metadata'], curr_slc_idx=_curr_slc_idx)

if __name__ == "__main__":
    main()
