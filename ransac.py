import numpy as np
def normalize_vector(v):
    norm = np.hypot(v[0], v[1])
    if norm == 0:
        return None
    return (v[0] / norm, v[1] / norm)


def angle_between(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    det = v1[0]*v2[1] - v1[1]*v2[0]
    return np.arctan2(det, dot)

def direction_vector(line):
    x1, y1, x2, y2 = line[0]
    return (x2 - x1, y2 - y1)

def ransac_line_clusters(lines, angle_threshold=np.deg2rad(10), max_clusters=3):
    clusters = []
    used = [False] * len(lines)

    for i, line in enumerate(lines):
        if used[i]:
            continue
        base_vec = direction_vector(line)
        base_vec = normalize_vector(base_vec)
        if base_vec is None:
            continue

        cluster = [line]
        used[i] = True

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            compare_vec = direction_vector(lines[j])
            compare_vec = normalize_vector(compare_vec)
            if compare_vec is None:
                continue
            angle = abs(angle_between(base_vec, compare_vec))
            if angle < angle_threshold or abs(angle - np.pi) < angle_threshold:
                cluster.append(lines[j])
                used[j] = True

        clusters.append(cluster)
        if len(clusters) >= max_clusters:
            break

    return clusters
