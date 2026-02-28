"""
airport_runway_detection.py
============================
基于ADS-B轨迹数据的机场与跑道识别算法。

核心逻辑：
  1. 提取每条轨迹最后5分钟内、绝对高度 < 1000m 的点作为候选着陆段
  2. 在候选着陆段内，结合速度骤降（方案A）与高度极低（方案D）定位接地点
     - 接地点用于表示降落位置坐标
     - 进近方向从接地点之前的进近段（高度 200~1000m）单独计算，避免滑行段污染
  3. 两阶段聚类：
     - 第一阶段：DBSCAN 空间聚类（默认 5km），将同一机场的所有降落聚为一组
     - 第二阶段：在每个空间聚类内按进近方向（30°）分裂，识别不同跑道
  4. 输出每个聚类（候选跑道）的中心坐标、降落次数、主方向、跑道编号
  5. 可视化到 folium 地图

使用示例（在 notebook 中）：
    import importlib, airport_runway_detection
    importlib.reload(airport_runway_detection)
    from airport_runway_detection import detect_airports, visualize_airports

    clusters, landing_events = detect_airports(df_clean)
    m = visualize_airports(clusters, landing_events, map_center=[23.16, 113.26])
    m
"""

import numpy as np
import pandas as pd
import folium
from math import radians, sin, cos, sqrt, atan2, degrees
from typing import Optional
from sklearn.cluster import DBSCAN


# ──────────────────────────────────────────────
# 基础几何工具函数
# ──────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点间的 Haversine 大圆距离，单位 km。"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算从 (lat1, lon1) 到 (lat2, lon2) 的真方位角，范围 [0, 360)。"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    return (degrees(atan2(x, y)) + 360) % 360


def circular_mean(angles: np.ndarray) -> float:
    """计算一组角度的循环均值，范围 [0, 360)。"""
    rad = np.radians(angles)
    return (degrees(atan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) + 360) % 360


def runway_angular_diff(a1: float, a2: float) -> float:
    """
    计算两个方向角的最小差值，范围 [0, 90]。
    跑道两端方向相差 180°，视为同一跑道方向。
    """
    diff = abs(a1 - a2) % 360
    if diff > 180:
        diff = 360 - diff
    if diff > 90:
        diff = 180 - diff
    return diff


def offset_point(lat: float, lng: float, bearing_deg: float, dist_km: float):
    """从 (lat, lng) 沿 bearing_deg 方向偏移 dist_km，返回新坐标，用于绘制跑道线。"""
    R = 6371.0
    d = dist_km / R
    b = radians(bearing_deg)
    lat1 = radians(lat)
    lon1 = radians(lng)
    lat2 = atan2(
        sin(lat1) * cos(d) + cos(lat1) * sin(d) * cos(b),
        sqrt((cos(lat1) * cos(d) - sin(lat1) * sin(d) * cos(b)) ** 2
             + (sin(d) * sin(b)) ** 2)
    )
    lon2 = lon1 + atan2(
        sin(b) * sin(d) * cos(lat1),
        cos(d) - sin(lat1) * sin(lat2)
    )
    return degrees(lat2), degrees(lon2)


# ──────────────────────────────────────────────
# 接地点定位（方案A + 方案D）
# ──────────────────────────────────────────────

def find_touchdown_point(
    landing_seg: pd.DataFrame,
    alt_col: str,
    touchdown_alt_m: float,
    touchdown_speed_ratio: float,
) -> Optional[pd.Series]:
    """
    在着陆段内定位接地点，结合速度骤降（A）与高度极低（D）。

    策略优先级：
      1. A+D 联合：高度 < touchdown_alt_m 且 速度 < 进近段最大速度 * ratio，取最早满足的点
      2. 纯D：取高度绝对值最低的点
      3. 兜底：返回着陆段最后一个点

    参数
    ----
    landing_seg           : 已按时间排序的着陆段 DataFrame
    alt_col               : 高度列名
    touchdown_alt_m       : 接地高度阈值（m）
    touchdown_speed_ratio : 接地速度 / 着陆段最大速度 的上限比值
    """
    if len(landing_seg) == 0:
        return None

    max_speed = landing_seg["speed"].max() if "speed" in landing_seg.columns else None
    low_alt_mask = landing_seg[alt_col].abs() < touchdown_alt_m

    # ── 策略1：A+D 联合 ──
    if low_alt_mask.any() and max_speed and max_speed > 0:
        low_alt_seg = landing_seg[low_alt_mask]
        speed_mask = low_alt_seg["speed"] < max_speed * touchdown_speed_ratio
        if speed_mask.any():
            return low_alt_seg[speed_mask].iloc[0]

    # ── 策略2：纯D，取最低高度点 ──
    if low_alt_mask.any():
        return landing_seg.loc[landing_seg[alt_col].abs().idxmin()]

    # ── 策略3：兜底 ──
    return landing_seg.iloc[-1]


# ──────────────────────────────────────────────
# 进近方向计算（接地点之前的进近段）
# ──────────────────────────────────────────────

def compute_approach_bearing(
    traj: pd.DataFrame,
    touchdown_time,
    time_col: str,
    alt_col: str,
    approach_alt_min_m: float,
    approach_alt_max_m: float,
    min_points: int,
) -> Optional[float]:
    """
    从接地点之前的进近段计算进近方向角。

    核心逻辑：
      - 只取时间上早于接地点的点（排除接地后滑行段）
      - 高度只设上限（< approach_alt_max_m），排除高空巡航段
      - 不设高度下限：飞机在最终进近阶段高度从高连续降至0，
        接地前的低空点（< 200m）恰恰是方向最准确的部分，不应过滤

    参数
    ----
    traj               : 完整轨迹 DataFrame（已按时间排序）
    touchdown_time     : 接地点时间戳，只取此时刻之前的点
    approach_alt_min_m : 保留参数（暂未使用，预留扩展）
    approach_alt_max_m : 进近段高度上限（m），排除高空巡航段
    min_points         : 至少需要多少个进近段点
    """
    # 取接地点之前、高度低于巡航高度的点
    approach = traj[
        (traj[time_col] < touchdown_time) &
        (traj[alt_col].abs() < approach_alt_max_m)
    ].copy()

    if len(approach) < 2:
        return None

    # 用进近段最后若干点计算连续方位角均值
    bearings_list = []
    rows = approach.tail(min_points * 3).reset_index(drop=True)
    for i in range(1, len(rows)):
        prev, curr = rows.iloc[i - 1], rows.iloc[i]
        if (abs(curr["lat"] - prev["lat"]) > 1e-7 or
                abs(curr["lng"] - prev["lng"]) > 1e-7):
            b = compute_bearing(prev["lat"], prev["lng"],
                                curr["lat"], curr["lng"])
            bearings_list.append(b)
        if len(bearings_list) >= min_points:
            break

    if len(bearings_list) == 0:
        return None

    return circular_mean(np.array(bearings_list))


# ──────────────────────────────────────────────
# 第一步：提取着陆事件
# ──────────────────────────────────────────────

def extract_landing_events(
    df: pd.DataFrame,
    time_col: str = "position_time",
    alt_col: str = "altitude",
    # 着陆段提取
    last_minutes: int = 5,
    alt_threshold_m: float = 1000.0,
    # 接地点定位（A+D）
    touchdown_alt_m: float = 50.0,
    touchdown_speed_ratio: float = 0.4,
    # 进近方向计算
    approach_alt_min_m: float = 200.0,
    approach_alt_max_m: float = 1000.0,
    min_points_for_bearing: int = 5,
    # 轨迹过滤
    final_alt_threshold_m: float = 200.0,
    min_speed_drop_ratio: float = 0.4,
    min_traj_points: int = 10,
    min_traj_minutes: float = 5.0,
) -> pd.DataFrame:
    """
    从已分割轨迹的 DataFrame 中提取着陆事件。

    接地点定位（A+D）：
      优先：高度 < touchdown_alt_m 且 速度 < 进近最大速度 * touchdown_speed_ratio
      退而：取高度绝对值最低的点
      兜底：着陆段最后一个点

    进近方向计算：
      专从接地点之前 approach_alt_min_m ~ approach_alt_max_m 高度范围的进近段计算，
      避免接地后滑行段污染方向角。

    参数
    ----
    df                    : 含 trajectory_id 列的清洗后 DataFrame
    time_col              : 时间列名（已转为 datetime 格式）
    alt_col               : 高度列名，单位 m，取绝对值
    last_minutes          : 着陆段末尾时间窗口（分钟）
    alt_threshold_m       : 着陆段高度上限（m）
    touchdown_alt_m       : 接地高度阈值（m），方案D
    touchdown_speed_ratio : 接地速度比阈值，方案A（0~1）
    approach_alt_min_m    : 进近段高度下限（m），用于计算方向角
    approach_alt_max_m    : 进近段高度上限（m），用于计算方向角
    min_points_for_bearing: 计算方向角所需最少点数
    final_alt_threshold_m : 接地点高度上限（m），超出则排除
    min_speed_drop_ratio  : 末尾速度 / 最大速度 上限，超出则排除
    min_traj_points       : 轨迹最少点数
    min_traj_minutes      : 轨迹最短时长（分钟）

    返回
    ----
    DataFrame，列：trajectory_id, icao, lat, lng, altitude, time, bearing
    """
    records = []

    for traj_id, traj in df.groupby("trajectory_id", sort=False):
        traj = traj.sort_values(time_col).reset_index(drop=True)

        # ── 过滤1：轨迹太短 ──
        traj_duration = (
            (traj[time_col].max() - traj[time_col].min()).total_seconds() / 60
        )
        if len(traj) < min_traj_points or traj_duration < min_traj_minutes:
            continue

        last_pt = traj.iloc[-1]

        # ── 过滤2：轨迹末尾高度上限 ──
        if abs(last_pt[alt_col]) > final_alt_threshold_m:
            continue

        # ── 过滤3：末尾速度应已显著降低 ──
        if "speed" in traj.columns:
            max_speed = traj["speed"].max()
            if max_speed > 0 and (last_pt["speed"] / max_speed) > min_speed_drop_ratio:
                continue

        # ── 提取着陆段（最后 N 分钟内高度 < alt_threshold_m 的点）──
        t_max = traj[time_col].max()
        t_cutoff = t_max - pd.Timedelta(minutes=last_minutes)
        landing_seg = traj[
            (traj[time_col] >= t_cutoff) &
            (traj[alt_col].abs() < alt_threshold_m)
        ].copy()

        if len(landing_seg) == 0:
            continue

        # ── 定位接地点（A + D 联合）──
        td_pt = find_touchdown_point(
            landing_seg, alt_col, touchdown_alt_m, touchdown_speed_ratio
        )
        if td_pt is None:
            continue

        # 接地点高度二次检查
        if abs(td_pt[alt_col]) > final_alt_threshold_m:
            continue

        # ── 计算进近方向（接地点之前的进近段）──
        bearing = compute_approach_bearing(
            traj,
            touchdown_time=td_pt[time_col],
            time_col=time_col,
            alt_col=alt_col,
            approach_alt_min_m=approach_alt_min_m,
            approach_alt_max_m=approach_alt_max_m,
            min_points=min_points_for_bearing,
        )
        if bearing is None:
            continue

        records.append({
            "trajectory_id": traj_id,
            "icao":          last_pt.get("icao", None),
            "lat":           td_pt["lat"],
            "lng":           td_pt["lng"],
            "altitude":      td_pt[alt_col],
            "time":          td_pt[time_col],
            "bearing":       bearing,
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# 第二步：两阶段聚类
# ──────────────────────────────────────────────

def _spatial_dbscan(
    events: pd.DataFrame,
    spatial_threshold_km: float,
    min_samples: int,
) -> np.ndarray:
    """
    第一阶段：DBSCAN 空间聚类。
    使用 haversine 度量，eps 单位弧度（= km / 地球半径）。
    返回 cluster_labels 数组，-1 为噪声。
    """
    coords_rad = np.radians(events[["lat", "lng"]].values)
    eps_rad = spatial_threshold_km / 6371.0
    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",
    ).fit(coords_rad)
    return db.labels_


def _split_by_direction(
    group: pd.DataFrame,
    angle_threshold_deg: float,
) -> np.ndarray:
    """
    第二阶段：在同一空间聚类内，按进近方向（angle_threshold_deg）分裂为跑道子聚类。

    算法：
      - 将方向折叠到 [0, 180) 以处理跑道对称性（如 RWY07 和 RWY25 属同一跑道）
      - 按折叠方向排序后贪心分配，减少对遍历顺序的依赖
      - 维护每个子聚类的循环均值方向，动态更新

    返回局部子聚类 ID 数组（0, 1, 2, ...）。
    """
    n = len(group)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    bearings = group["bearing"].values
    # 折叠到 [0, 180)：跑道两端视为同一方向
    folded = bearings % 180
    order = np.argsort(folded)

    sub_labels = np.full(n, -1, dtype=int)
    sub_centers = []  # 每个子聚类的折叠方向均值

    sub_id = 0
    for idx in order:
        b_fold = folded[idx]
        assigned = False
        for sid, center in enumerate(sub_centers):
            diff = abs(b_fold - center) % 180
            if diff > 90:
                diff = 180 - diff
            if diff <= angle_threshold_deg:
                sub_labels[idx] = sid
                # 更新子聚类折叠方向均值
                members = folded[sub_labels == sid]
                rad2 = np.radians(members * 2)  # ×2 映射到 [0,360) 做循环均值
                sub_centers[sid] = (
                    (degrees(atan2(np.mean(np.sin(rad2)),
                                  np.mean(np.cos(rad2)))) + 360) % 360
                ) / 2
                assigned = True
                break
        if not assigned:
            sub_labels[idx] = sub_id
            sub_centers.append(b_fold)
            sub_id += 1

    return sub_labels


def cluster_landing_events(
    events: pd.DataFrame,
    spatial_threshold_km: float = 5.0,
    angle_threshold_deg: float = 30.0,
    min_samples: int = 2,
) -> pd.DataFrame:
    """
    对着陆事件进行两阶段聚类。

    第一阶段（机场级）：DBSCAN 空间聚类
      - eps = spatial_threshold_km（默认 5km）
      - 噪声点（airport_id = -1）不参与第二阶段

    第二阶段（跑道级）：方向分裂
      - 在每个空间聚类内按进近方向（30°）拆分
      - 每个子聚类对应一条跑道

    cluster_id 编码规则：airport_id * 100 + runway_sub_id（全局唯一）

    参数
    ----
    events               : extract_landing_events 的输出
    spatial_threshold_km : 第一阶段空间聚类半径（km）
    angle_threshold_deg  : 第二阶段方向分裂阈值（°）
    min_samples          : DBSCAN min_samples

    返回
    ----
    原 events DataFrame 加上 cluster_id, airport_id, runway_id 列
    """
    if len(events) == 0:
        return events.copy()

    ev = events.copy().reset_index(drop=True)

    # ── 第一阶段：空间 DBSCAN ──
    ev["airport_id"] = _spatial_dbscan(ev, spatial_threshold_km, min_samples)

    # ── 第二阶段：方向分裂 ──
    ev["runway_id"]  = -1
    ev["cluster_id"] = -1

    for ap_id in sorted(ev["airport_id"].unique()):
        if ap_id == -1:
            continue
        mask = ev["airport_id"] == ap_id
        group = ev[mask].copy()
        sub_labels = _split_by_direction(group, angle_threshold_deg)
        ev.loc[mask, "runway_id"]  = sub_labels
        ev.loc[mask, "cluster_id"] = ap_id * 100 + sub_labels

    return ev


# ──────────────────────────────────────────────
# 第三步：汇总聚类为机场/跑道信息
# ──────────────────────────────────────────────

def summarize_clusters(events_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    将聚类后的着陆事件汇总为机场/跑道信息表。

    返回 DataFrame，列：
        cluster_id, airport_id, runway_id,
        center_lat, center_lng, landing_count,
        main_bearing, runway_heading_1, runway_heading_2, runway_name
    """
    rows = []
    for cid, grp in events_clustered.groupby("cluster_id"):
        if cid == -1:
            continue
        mb = circular_mean(grp["bearing"].values)
        rwy1 = mb % 360
        rwy2 = (mb + 180) % 360
        rwy_num1 = int(round(rwy1 / 10)) % 36 or 36
        rwy_num2 = int(round(rwy2 / 10)) % 36 or 36

        rows.append({
            "cluster_id":       int(cid),
            "airport_id":       int(grp["airport_id"].iloc[0]),
            "runway_id":        int(grp["runway_id"].iloc[0]),
            "center_lat":       grp["lat"].mean(),
            "center_lng":       grp["lng"].mean(),
            "landing_count":    len(grp),
            "main_bearing":     round(mb, 1),
            "runway_heading_1": round(rwy1, 1),
            "runway_heading_2": round(rwy2, 1),
            "runway_name":      f"RWY {rwy_num1:02d}/{rwy_num2:02d}",
        })

    df_clusters = pd.DataFrame(rows)
    if len(df_clusters) > 0:
        df_clusters = df_clusters.sort_values(
            "landing_count", ascending=False
        ).reset_index(drop=True)
    return df_clusters


# ──────────────────────────────────────────────
# 主入口函数
# ──────────────────────────────────────────────

def detect_airports(
    df_clean: pd.DataFrame,
    time_col: str = "position_time",
    alt_col: str = "altitude",
    # 着陆段提取
    last_minutes: int = 5,
    alt_threshold_m: float = 1000.0,
    # 接地点定位（A+D）
    touchdown_alt_m: float = 50.0,
    touchdown_speed_ratio: float = 0.4,
    # 进近方向计算
    approach_alt_min_m: float = 200.0,
    approach_alt_max_m: float = 1000.0,
    min_points_for_bearing: int = 5,
    # 轨迹过滤
    final_alt_threshold_m: float = 200.0,
    min_speed_drop_ratio: float = 0.4,
    min_traj_points: int = 10,
    min_traj_minutes: float = 5.0,
    # 两阶段聚类
    spatial_threshold_km: float = 5.0,
    angle_threshold_deg: float = 30.0,
    min_samples: int = 2,
    # 结果过滤
    min_landings: int = 2,
):
    """
    机场跑道识别主函数（两阶段聚类版）。

    参数
    ----
    df_clean               : 已分割轨迹、含 trajectory_id 列的 DataFrame
    time_col               : 时间列（datetime 格式）
    alt_col                : 高度列，单位 m，取绝对值
    last_minutes           : 着陆段末尾时间窗口（分钟）
    alt_threshold_m        : 着陆段高度上限（m）
    touchdown_alt_m        : 接地高度阈值（m），方案D
    touchdown_speed_ratio  : 接地速度比阈值，方案A（0~1）
    approach_alt_min_m     : 进近段高度下限（m），用于计算方向角
    approach_alt_max_m     : 进近段高度上限（m），用于计算方向角
    min_points_for_bearing : 计算方向角所需最少点数
    final_alt_threshold_m  : 接地点高度上限（m），超出则排除
    min_speed_drop_ratio   : 末尾速度比上限，超出则排除
    min_traj_points        : 轨迹最少点数
    min_traj_minutes       : 轨迹最短时长（分钟）
    spatial_threshold_km   : 第一阶段空间聚类半径（km）
    angle_threshold_deg    : 第二阶段方向分裂阈值（°）
    min_samples            : DBSCAN 最少样本数
    min_landings           : 保留聚类的最少降落次数

    返回
    ----
    clusters        : DataFrame，每行为一条候选跑道
    landing_events  : DataFrame，所有着陆事件（含 cluster_id, airport_id, runway_id）
    """
    print("=" * 62)
    print("  机场 / 跑道识别算法（A+D 接地点 + 两阶段聚类）")
    print("=" * 62)
    print(f"  轨迹总数            : {df_clean['trajectory_id'].nunique():,}")
    print(f"  着陆段时间窗口       : 最后 {last_minutes} 分钟")
    print(f"  着陆段高度上限       : {alt_threshold_m} m")
    print(f"  接地高度阈值（D）    : {touchdown_alt_m} m")
    print(f"  接地速度比（A）      : {touchdown_speed_ratio}")
    print(f"  进近段高度范围       : {approach_alt_min_m} ~ {approach_alt_max_m} m")
    print(f"  轨迹最少点/时长      : {min_traj_points} 点 / {min_traj_minutes} 分钟")
    print(f"  第一阶段空间阈值     : {spatial_threshold_km} km")
    print(f"  第二阶段方向阈值     : {angle_threshold_deg} °")
    print()

    # Step 1: 提取着陆事件
    print("▶ Step 1: 提取着陆事件（A+D 接地点定位）...")
    landing_events = extract_landing_events(
        df_clean,
        time_col=time_col, alt_col=alt_col,
        last_minutes=last_minutes, alt_threshold_m=alt_threshold_m,
        touchdown_alt_m=touchdown_alt_m,
        touchdown_speed_ratio=touchdown_speed_ratio,
        approach_alt_min_m=approach_alt_min_m,
        approach_alt_max_m=approach_alt_max_m,
        min_points_for_bearing=min_points_for_bearing,
        final_alt_threshold_m=final_alt_threshold_m,
        min_speed_drop_ratio=min_speed_drop_ratio,
        min_traj_points=min_traj_points,
        min_traj_minutes=min_traj_minutes,
    )
    print(f"  → 提取到 {len(landing_events)} 个着陆事件")

    if len(landing_events) == 0:
        print("  ⚠ 未发现任何着陆事件，请检查参数。")
        return pd.DataFrame(), pd.DataFrame()

    # Step 2: 两阶段聚类
    print("▶ Step 2: 两阶段聚类（空间 DBSCAN → 方向分裂）...")
    events_clustered = cluster_landing_events(
        landing_events,
        spatial_threshold_km=spatial_threshold_km,
        angle_threshold_deg=angle_threshold_deg,
        min_samples=min_samples,
    )
    n_airports = events_clustered[
        events_clustered["airport_id"] >= 0
    ]["airport_id"].nunique()
    n_noise = (events_clustered["airport_id"] == -1).sum()
    n_runways = events_clustered[
        events_clustered["cluster_id"] >= 0
    ]["cluster_id"].nunique()
    print(f"  → 第一阶段：{n_airports} 个空间聚类（机场候选），噪声点 {n_noise} 个")
    print(f"  → 第二阶段：方向分裂后共 {n_runways} 条候选跑道")

    # Step 3: 汇总并过滤
    print("▶ Step 3: 汇总聚类结果 ...")
    clusters = summarize_clusters(events_clustered)
    before = len(clusters)
    clusters = clusters[
        clusters["landing_count"] >= min_landings
    ].reset_index(drop=True)
    print(f"  → 过滤 min_landings={min_landings} 后保留 {len(clusters)} / {before} 条跑道")

    print()
    print(f"{'排名':<5} {'机场ID':<8} {'中心坐标':<28} {'降落次数':<10}"
          f"{'跑道名称':<15} {'主方位角':>8}")
    print("-" * 78)
    for rank, row in clusters.iterrows():
        coord = f"({row['center_lat']:.4f}, {row['center_lng']:.4f})"
        print(f"  {rank+1:<4} AP{int(row['airport_id']):<6} {coord:<28}"
              f"{int(row['landing_count']):<10} {row['runway_name']:<15}"
              f"{row['main_bearing']:>7.1f}°")
    print()

    return clusters, events_clustered


# ──────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────

def visualize_airports(
    clusters: pd.DataFrame,
    landing_events: pd.DataFrame,
    map_center: Optional[list] = None,
    zoom_start: int = 9,
    runway_half_length_km: float = 1.5,
    show_landing_points: bool = True,
    output_path: str = "airport_runway_detection_map.html",
):
    """
    将机场识别结果可视化到 folium 地图。

    参数
    ----
    clusters             : detect_airports 返回的 clusters DataFrame
    landing_events       : detect_airports 返回的 landing_events DataFrame
    map_center           : 地图中心 [lat, lng]，默认用着陆事件均值
    zoom_start           : folium 初始缩放等级
    runway_half_length_km: 跑道方向线的半长（km）
    show_landing_points  : 是否绘制单个着陆事件点
    output_path          : 输出 HTML 文件路径

    返回
    ----
    m : folium.Map 对象（在 notebook 中可直接显示）
    """
    if len(clusters) == 0:
        print("⚠ 无聚类结果可视化。")
        return None

    if map_center is None:
        map_center = [landing_events["lat"].mean(), landing_events["lng"].mean()]

    m = folium.Map(location=map_center, zoom_start=zoom_start,
                   tiles="CartoDB positron")

    max_count = clusters["landing_count"].max()

    def get_color(count):
        if count >= max_count * 0.5:
            return "#d73027"
        elif count >= max_count * 0.15:
            return "#fc8d59"
        else:
            return "#4575b4"

    # ── 着陆事件点 ──
    if show_landing_points and len(landing_events) > 0:
        landing_group = folium.FeatureGroup(name="着陆事件点", show=True)
        for _, ev in landing_events.iterrows():
            folium.CircleMarker(
                location=[ev["lat"], ev["lng"]],
                radius=3,
                color="#999999", fill=True,
                fill_color="#cccccc", fill_opacity=0.5, weight=1,
                popup=folium.Popup(
                    f"<b>轨迹</b>: {ev['trajectory_id']}<br>"
                    f"<b>ICAO</b>: {ev.get('icao', 'N/A')}<br>"
                    f"<b>进近方向</b>: {ev['bearing']:.1f}°<br>"
                    f"<b>高度</b>: {ev['altitude']:.0f} m<br>"
                    f"<b>机场ID</b>: AP{int(ev.get('airport_id', -1))}<br>"
                    f"<b>聚类ID</b>: {int(ev.get('cluster_id', -1))}",
                    max_width=220
                )
            ).add_to(landing_group)
        landing_group.add_to(m)

    # ── 跑道聚类 ──
    airport_group = folium.FeatureGroup(name="识别机场/跑道", show=True)

    for rank, row in clusters.iterrows():
        lat, lng = row["center_lat"], row["center_lng"]
        count = row["landing_count"]
        color = get_color(count)
        bearing = row["main_bearing"]

        # 跑道方向线
        p1 = offset_point(lat, lng, bearing, runway_half_length_km)
        p2 = offset_point(lat, lng, (bearing + 180) % 360, runway_half_length_km)
        folium.PolyLine(
            locations=[p1, p2], color=color, weight=4, opacity=0.9,
            tooltip=f"{row['runway_name']}  |  {count}次降落"
        ).add_to(airport_group)

        # 中心圆（大小随降落次数变化）
        radius = max(8, min(22, 8 + count // 10))
        folium.CircleMarker(
            location=[lat, lng], radius=radius,
            color=color, fill=True, fill_color=color,
            fill_opacity=0.85, weight=2,
            popup=folium.Popup(
                f"<h4 style='margin:0'>候选跑道 #{rank+1}</h4>"
                f"<hr style='margin:4px 0'>"
                f"<b>机场 ID</b>: AP{int(row['airport_id'])}<br>"
                f"<b>跑道子ID</b>: {int(row['runway_id'])}<br>"
                f"<b>坐标</b>: {lat:.5f}, {lng:.5f}<br>"
                f"<b>降落次数</b>: {count}<br>"
                f"<b>跑道名称</b>: {row['runway_name']}<br>"
                f"<b>主方位角</b>: {bearing:.1f}°<br>"
                f"<b>跑道方向1</b>: {row['runway_heading_1']:.1f}°<br>"
                f"<b>跑道方向2</b>: {row['runway_heading_2']:.1f}°",
                max_width=240
            ),
            tooltip=(f"#{rank+1} AP{int(row['airport_id'])} "
                     f"{row['runway_name']} 降落{count}次")
        ).add_to(airport_group)

        # 编号标签
        folium.Marker(
            location=[lat, lng],
            icon=folium.DivIcon(
                html=(f'<div style="font-size:10px;font-weight:bold;color:white;'
                      f'background:{color};border-radius:50%;width:20px;height:20px;'
                      f'line-height:20px;text-align:center;'
                      f'margin-top:-10px;margin-left:-10px;">{rank+1}</div>'),
                icon_size=(20, 20), icon_anchor=(10, 10)
            )
        ).add_to(airport_group)

    airport_group.add_to(m)

    # ── 图例 ──
    n_airports = clusters["airport_id"].nunique()
    legend_html = f"""
    <div style="
        position:fixed;bottom:30px;left:30px;z-index:1000;
        background:white;padding:12px 16px;border-radius:8px;
        border:1px solid #ccc;font-size:13px;line-height:1.8;
        box-shadow:2px 2px 6px rgba(0,0,0,0.2);">
        <b>机场识别结果</b><br>
        识别机场 <b>{n_airports}</b> 个 / 跑道 <b>{len(clusters)}</b> 条<br>
        <hr style="margin:6px 0">
        <span style="color:#d73027">●</span> 主要机场（≥{int(max_count*0.5)} 次）<br>
        <span style="color:#fc8d59">●</span> 中型机场（≥{int(max_count*0.15)} 次）<br>
        <span style="color:#4575b4">●</span> 小型/通航机场<br>
        <span style="color:#999">●</span> 着陆事件点<br>
        <hr style="margin:6px 0">
        <i>线段方向 = 跑道进近方向</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    m.save(output_path)
    print(f"✓ 地图已保存至: {output_path}")
    return m
