import geopandas as gpd
import osmnx as ox
from shapely.ops import substring
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString
import time
import os


def process_chunk(df_chunk, crs, G, G_base, max_dist, buffer_dist):
    pid = os.getpid()
    start = time.time()
    out = []

    for i, (idx, row) in enumerate(df_chunk.iterrows(), start=1):

        if i % 10 == 0:
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            print(
                f"[PID {pid}] {i}/{len(df_chunk)} rows | "
                f"{elapsed:.1f}s elapsed | {rate:.2f} rows/s"
            )

        try:
            reach, buffer_geom = compute_reach_and_buffer(
                row, crs, G, G_base,
                max_dist=max_dist,
                buffer_dist=buffer_dist
            )
        except Exception:
            reach, buffer_geom = None, None

        out.append((idx, reach, buffer_geom))

    total = time.time() - start
    print(f"[PID {pid}] chunk done in {total:.1f}s")

    return out


def compute_reach_and_buffer(row, property_crs, G, G_base, max_dist=200, buffer_dist=10):
    try:
        reach = isodistance_from_access_point(
            row,
            property_crs,
            G,
            G_base,
            max_dist=max_dist
        )

        if reach is None or reach.is_empty:
            return MultiLineString([]), None

        reach_buffer = reach.buffer(buffer_dist)

        return reach, reach_buffer

    except Exception as e:
        print(f"Error for property {row.name}: {e}")
        return MultiLineString([]), None


MIN_SEGMENT_LEN = 0.5  # metres


def is_valid_segment(g):
    return (
        isinstance(g, LineString)
        and not g.is_empty
        and g.length > MIN_SEGMENT_LEN
    )


def isodistance_from_access_point(property_row, property_crs, G, G_base, max_dist=200):

    access_pt = property_row['access_point']

    access_pt_proj = gpd.GeoSeries(
        [access_pt], crs=property_crs).to_crs("EPSG:2193").iloc[0]

    # Find nearest edge efficiently using spatial index (front street)
    u, v, k = ox.distance.nearest_edges(
        G, access_pt_proj.x, access_pt_proj.y, return_dist=False)

    data = G.edges[u, v, k]

    if 'geometry' in data:
        edge_geom = data['geometry']
    else:
        # Create geometry from node coordinates
        u_pt = Point(G.nodes[u]['x'], G.nodes[u]['y'])
        v_pt = Point(G.nodes[v]['x'], G.nodes[v]['y'])
        edge_geom = LineString([u_pt, v_pt])

    edge_length = data['length']

    proj_dist = edge_geom.project(access_pt_proj)
    proj_dist = min(max(proj_dist, 0), edge_length)

    dist_to_u = proj_dist
    dist_to_v = edge_length - proj_dist

    needs_split = not (dist_to_u < 0.5 or dist_to_v < 0.5)

    if needs_split:
        G_work = G_base.copy()
    else:
        G_work = G_base

    if dist_to_u < 0.5:
        start_node = u

    elif dist_to_v < 0.5:
        start_node = v

    else:
        access_node = f"access_{property_row.name}"
        G_work.add_node(access_node, x=access_pt_proj.x, y=access_pt_proj.y)

        geom_to_u = substring(edge_geom, 0, proj_dist)
        geom_to_v = substring(edge_geom, proj_dist, edge_length)

        G_work.add_edge(access_node, u, length=dist_to_u, geometry=geom_to_u)
        G_work.add_edge(access_node, v, length=dist_to_v, geometry=geom_to_v)

        if G_work.has_edge(u, v):
            G_work.remove_edge(u, v, k)

        start_node = access_node

    lengths = nx.single_source_dijkstra_path_length(
        G_work,
        start_node,
        cutoff=max_dist,
        weight='length'
    )

    reachable = set(lengths.keys())

    lines = []

    for u_node, v_node, data in G_work.edges(data=True):

        if u_node not in reachable and v_node not in reachable:
            continue

        du = lengths.get(u_node, float('inf'))
        dv = lengths.get(v_node, float('inf'))

        if du > max_dist and dv > max_dist:
            continue

        geom = data.get('geometry')

        if geom is None:
            u_pt = Point(G_work.nodes[u_node]['x'], G_work.nodes[u_node]['y'])
            v_pt = Point(G_work.nodes[v_node]['x'], G_work.nodes[v_node]['y'])
            geom = LineString([u_pt, v_pt])

        edge_len = data.get('length', geom.length)

        if du <= max_dist and dv <= max_dist:
            lines.append(geom)

        elif du <= max_dist < dv:
            remaining = max_dist - du
            if remaining > MIN_SEGMENT_LEN:
                u_pt = Point(G_work.nodes[u_node]['x'],
                             G_work.nodes[u_node]['y'])
                v_pt = Point(G_work.nodes[v_node]['x'],
                             G_work.nodes[v_node]['y'])

                geom_start = Point(geom.coords[0])

                if geom_start.distance(u_pt) < geom_start.distance(v_pt):
                    clipped = substring(geom, 0, min(remaining, edge_len))
                else:
                    start_dist = max(0, edge_len - remaining)
                    clipped = substring(geom, start_dist, edge_len)

                if is_valid_segment(clipped):
                    lines.append(clipped)

        elif dv <= max_dist < du:
            remaining = max_dist - dv
            if remaining > MIN_SEGMENT_LEN:
                u_pt = Point(G_work.nodes[u_node]['x'],
                             G_work.nodes[u_node]['y'])
                v_pt = Point(G_work.nodes[v_node]['x'],
                             G_work.nodes[v_node]['y'])

                geom_start = Point(geom.coords[0])

                if geom_start.distance(v_pt) < geom_start.distance(u_pt):
                    clipped = substring(geom, 0, min(remaining, edge_len))
                else:
                    start_dist = max(0, edge_len - remaining)
                    clipped = substring(geom, start_dist, edge_len)

                if is_valid_segment(clipped):
                    lines.append(clipped)

    if not lines:
        return MultiLineString([])

    result = MultiLineString(lines)
    return gpd.GeoSeries([result], crs=G.graph['crs']).to_crs(property_crs).iloc[0]
