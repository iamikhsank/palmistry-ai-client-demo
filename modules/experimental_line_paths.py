"""Exploratory line tracing. Labels are spatial hypotheses, not validated predictions."""
import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra

from modules.line_geometry import thin


def trace_components(mask, min_pixels=20, max_nodes=5000, max_endpoints=24):
    """Trace the longest shortest endpoint-to-endpoint path in each skeleton.

    Return image-relative coordinates and lengths in pixels. A path through a
    crossing can switch lines; junctions are candidates, not branch counts.
    Large or highly tangled components are skipped instead of guessed.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) < 3:
        raise ValueError('Expected a 2D mask')
    skel = thin(mask)
    labels, n = ndimage.label(skel, structure=np.ones((3, 3)))
    paths = []
    for label in range(1, n + 1):
        ys, xs = np.nonzero(labels == label)
        count = len(xs)
        if count < min_pixels or count > max_nodes:
            continue
        coords = list(zip(ys.tolist(), xs.tolist()))
        indices = {p: i for i, p in enumerate(coords)}
        rows, cols, weights = [], [], []
        for i, (y, x) in enumerate(coords):
            for dy, dx in ((0,1),(1,0),(1,1),(1,-1)):
                j = indices.get((y+dy, x+dx))
                if j is None:
                    continue
                # Do not add diagonal shortcuts around an existing right angle.
                if dy and dx and ((y, x+dx) in indices or (y+dy, x) in indices):
                    continue
                rows.extend((i,j)); cols.extend((j,i))
                weights.extend((float(np.hypot(dx,dy)),) * 2)
        graph = sparse.csr_matrix((weights, (rows, cols)), shape=(count,count))
        degree = np.diff(graph.indptr)
        ends = np.flatnonzero(degree == 1)
        if len(ends) < 2 or len(ends) > max_endpoints:
            continue
        distances, previous = dijkstra(graph, directed=False, indices=ends, return_predecessors=True)
        pair = distances[:,ends]
        pair[~np.isfinite(pair)] = -1
        a, b = np.unravel_index(np.argmax(pair), pair.shape)
        length = float(pair[a,b])
        if length <= 0:
            continue
        node, start = int(ends[b]), int(ends[a])
        path = [node]
        while node != start:
            node = int(previous[a,node])
            if node < 0 or len(path) > count:
                raise ValueError('Invalid skeleton path')
            path.append(node)
        xy = np.array([(xs[i], ys[i]) for i in reversed(path)], dtype=float)
        junctions = (degree >= 3).sum()
        paths.append({'xy': xy, 'length_px': length, 'junction_pixels': int(junctions)})
    return paths


def palm_coordinates(xy, landmarks, size):
    """Transform pixels to coordinates from index MCP toward little MCP and wrist."""
    landmarks = np.asarray(landmarks, dtype=float)
    if landmarks.shape != (21,2) or not np.isfinite(landmarks).all() or size < 3:
        raise ValueError('Invalid landmarks')
    index, little, wrist = landmarks[[5,17,0]]
    matrix = np.column_stack((little-index, wrist-(index+little)/2))
    if abs(np.linalg.det(matrix)) < 0.01:
        raise ValueError('Palm landmarks are degenerate')
    return np.linalg.solve(matrix, (np.asarray(xy).T / (size-1) - index[:,None])).T


def propose_line_zones(paths, landmarks, size):
    """Rank geometric hypotheses in broad reference zones; return no confidence.

    These rules are not trained or validated. Ambiguous or overlapping candidates
    are omitted. Returned lengths are relative to MCP width, not physical units.
    """
    landmarks = np.asarray(landmarks, dtype=float)
    if landmarks.shape != (21,2):
        raise ValueError('Expected 21 landmarks')
    width = float(np.linalg.norm(landmarks[17]-landmarks[5]) * (size-1))
    if width < 5:
        raise ValueError('Palm width is too small')
    ranked = {'asmara': [], 'kepala': [], 'kehidupan': []}
    for i, item in enumerate(paths):
        uv = palm_coordinates(item['xy'], landmarks, size)
        u, v = uv[:,0], uv[:,1]
        du, dv = float(np.ptp(u)), float(np.ptp(v))
        mean_u, mean_v = float(np.mean(u)), float(np.mean(v))
        relative_length = item['length_px'] / width
        if relative_length < .25:
            continue
        options = []
        if .05 < mean_v < .43 and du > .25 and du > dv:
            options.append(('asmara', abs(mean_v-.24)))
        if .38 < mean_v < .75 and du > .25 and du > dv:
            options.append(('kepala', abs(mean_v-.55)))
        if -.2 < mean_u < .48 and .25 < mean_v < .9 and dv > .3 and dv > du:
            options.append(('kehidupan', abs(mean_u-.2)))
        for label, score in options:
            ranked[label].append((score, i, relative_length, (mean_u, mean_v)))
    result = {}
    for label, candidates in ranked.items():
        candidates.sort()
        if not candidates or (len(candidates) > 1 and candidates[1][0]-candidates[0][0] < .08):
            continue
        score, i, length, position = candidates[0]
        result[label] = {'path_index': i, 'length_palm_widths': length,
                         'position_palm': position,
                         'junction_pixels': paths[i]['junction_pixels']}
    # One connected path must never be reported as multiple distinct lines.
    overlapping = {v['path_index'] for v in result.values()
                   if sum(other['path_index'] == v['path_index'] for other in result.values()) > 1}
    for label in list(result):
        if result[label]['path_index'] in overlapping:
            result.pop(label)
    return result


def trace_ridge_zones(ridge, landmarks, palm_mask, size=256, percentile=85):
    """Generate exploratory paths and spatial hypotheses within a palm mask."""
    ridge = np.asarray(ridge)
    mask = np.asarray(palm_mask, dtype=bool)
    if (ridge.ndim != 2 or not np.isfinite(ridge).all() or mask.shape != (size,size)
            or not mask.any() or not 0 <= percentile <= 100):
        raise ValueError('Invalid ridge or palm mask')
    ridge = np.asarray(Image.fromarray(np.clip(ridge,0,255).astype(np.uint8)).resize(
        (size,size), Image.Resampling.BILINEAR))
    cutoff = max(25., float(np.percentile(ridge[mask], percentile)))
    paths = trace_components((ridge > cutoff) & mask)
    return paths, propose_line_zones(paths, landmarks, size)
