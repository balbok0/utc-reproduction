import os
import sys
from numpy.lib.npyio import save
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import sumolib
import numpy as np
import argparse
from pathlib import Path
import subprocess


def generate_rou_xml_files(
    network_file: str,
    num_steps: int = 14000,
    delta_step: int = 2,
    peak_step_start: int = 0,
    peak_step_end: int = 0,
    bootstrap: int = 1,
    prefix: str = None,
    rand_path_prob: float = 0.4,
    rand_path_len: int = 7,
):
    net = sumolib.net.readNet(network_file)

    save_folder = Path(network_file).parent / "routes"
    if not save_folder.exists():
        os.makedirs(str(save_folder))
    prefix = Path(network_file).name[:-8] if prefix is None else prefix

    outgoing_edges = [edge for edge in net.getEdges() if len(edge.getOutgoing()) == 0]
    incoming_edges = [edge for edge in net.getEdges() if len(edge.getIncoming()) == 0]

    if bootstrap == 1:
        __generate_single_rou_xml_file(
            net=net,
            outgoing_edges=outgoing_edges,
            incoming_edges=incoming_edges,
            out_file_name=save_folder / f"{prefix}.rou.xml",
            num_steps=num_steps,
            delta_step=delta_step,
            peak_step_start=peak_step_start,
            peak_step_end=peak_step_end,
            rand_path_prob=rand_path_prob,
            rand_path_len=rand_path_len,
        )
    else:
        for b_idx in range(bootstrap):
            __generate_single_rou_xml_file(
                net=net,
                outgoing_edges=outgoing_edges,
                incoming_edges=incoming_edges,
                out_file_name=save_folder / f"{prefix}_{b_idx}.rou.xml",
                num_steps=num_steps,
                delta_step=delta_step,
                peak_step_start=peak_step_start,
                peak_step_end=peak_step_end,
                rand_path_prob=rand_path_prob,
                rand_path_len=rand_path_len,
            )


def __generate_single_rou_xml_file(
    net,
    incoming_edges,
    outgoing_edges,
    out_file_name,
    num_steps,
    delta_step,
    peak_step_start,
    peak_step_end,
    rand_path_prob,
    rand_path_len,
):
    paths = []
    curr_step = 0
    while curr_step < num_steps:
        in_edge = np.random.choice(incoming_edges)
        if np.random.random() < rand_path_prob:
            path = [in_edge]
            while len(path) < rand_path_len:
                out_from_curr = list(path[-1].getOutgoing().keys())
                if len(out_from_curr) == 0:
                    break
                path.append(np.random.choice(out_from_curr))
        else:
            out_edge = np.random.choice(outgoing_edges)
            path = net.getShortestPath(in_edge, out_edge)
            path = path[0]

        paths.append((path, curr_step))
        curr_step += delta_step / 2 if peak_step_start < curr_step < peak_step_end else delta_step

    xml_content = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
        "<!-- generated on Mon Dec 18 10:05:26 2017 by SUMO duarouter Version 0.30.0\n",
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
        "<configuration xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"http://sumo.dlr.de/xsd/duarouterConfiguration.xsd\">\n",
        "    <input>\n",
        "        <net-file value=\"networkFiles/3x3Grid2lanes.net.xml\"/>\n",
        "        <trip-files value=\"tripse14000.trips.xml\"/>\n",
        "    </input>\n",
        "    <output>\n",
        "        <output-file value=\"routese14000.rou.xml\"/>\n",
        "    </output>\n",
        "</configuration>\n",
        "-->\n",
        "<routes xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"http://sumo.dlr.de/xsd/routes_file.xsd\">\n",
    ]

    for vehicle_idx, path in enumerate(paths):
        edges = [e.getID() for e in path[0]]
        time_path = path[1]
        path_str = " ".join(edges)
        path_xml = [
            f"\t<vehicle id=\"{vehicle_idx}\" depart=\"{time_path:.2f}\">\n",
            f"\t\t<route edges=\"{path_str}\"/>\n",
            "\t</vehicle>\n",
        ]
        xml_content.extend(path_xml)

    xml_content.append("</routes>")

    with open(out_file_name, mode="w") as f:
        f.writelines(xml_content)


def demote():
    def set_ids():
        os.setuid(1000)
        os.setgid(1000)
    return set_ids


def generate_reproduce_rou_xml_files(network_file: str, bootstrap: int = 1, prefix: str = None, *args, **kwargs):
    net = sumolib.net.readNet(network_file)

    save_folder = Path(network_file).parent / "routes"
    if not save_folder.exists():
        os.makedirs(str(save_folder))
    prefix = Path(network_file).name[:-8] if prefix is None else prefix

    outgoing_edges = [edge.getID() for edge in net.getEdges() if len(edge.getOutgoing()) == 0]
    incoming_edges = [edge.getID() for edge in net.getEdges() if len(edge.getIncoming()) == 0]

    if bootstrap == 1:
        __generate_reproduce_single_rou_xml_file(
            network_file=str(network_file),
            outgoing_edges=outgoing_edges,
            incoming_edges=incoming_edges,
            out_file_prefix=str(save_folder / f"{prefix}"),
        )
    else:
        for b_idx in range(bootstrap):
            __generate_reproduce_single_rou_xml_file(
                network_file=str(network_file),
                outgoing_edges=outgoing_edges,
                incoming_edges=incoming_edges,
                out_file_prefix=str(save_folder / f"{prefix}_{b_idx}"),
            )


def __generate_reproduce_single_rou_xml_file(network_file, incoming_edges, outgoing_edges, out_file_prefix):
    in_lines = ["<edgedata>\n"]
    out_lines = ["<edgedata>\n"]
    # Loop over 15min intervals
    for start in range(0, 3600, 900):
        in_lines.append(
            f"\t<interval begin=\"{start}\" end=\"{start + 900}\"/>\n"
        )
        out_lines.append(
            f"\t<interval begin=\"{start}\" end=\"{start + 900}\"/>\n"
        )
        # Gaussian shifted to have all values at least slightly above 0
        in_edge_probs = np.random.normal(size=len(incoming_edges))
        min_in_edge_probs = np.min(in_edge_probs)
        in_edge_probs -= min_in_edge_probs * (1 - .1 * np.sign(min_in_edge_probs))

        out_edge_probs = np.random.normal(size=len(outgoing_edges))
        min_out_edge_probs = np.min(out_edge_probs)
        out_edge_probs -= min_out_edge_probs * (1 - .1 * np.sign(min_out_edge_probs))
        for edge, edge_prob in zip(incoming_edges, in_edge_probs):
            in_lines.append(
                f"\t\t<edge id=\"{edge}\" value=\"{edge_prob}\"/>\n"
            )
        for edge, edge_prob in zip(outgoing_edges, out_edge_probs):
            out_lines.append(
                f"\t\t<edge id=\"{edge}\" value=\"{edge_prob}\"/>\n"
            )
        in_lines.append("\t</interval>\n")
        out_lines.append("\t</interval>\n")
    in_lines.append("</edgedata>\n")
    out_lines.append("</edgedata>\n")

    with open(f"{out_file_prefix}.src.xml", mode="w") as f:
        f.writelines(in_lines)
    with open(f"{out_file_prefix}.dst.xml", mode="w") as f:
        f.writelines(out_lines)

    args = [
        os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py"),
        "--weights-prefix",
        "tmp_weights",
        "-n",
        network_file,
        "-r",
        f"{out_file_prefix}.rou.xml",
        "-p",
        f"{np.random.rand() * 1.9 + 0.1}",
        "--binomial",
        f"{np.random.randint(10, 51)}",
        "-e",
        f"{3600}",
    ]
    print(" ".join(args))
    subprocess.Popen(
        " ".join(args),
        preexec_fn=demote(),
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("generate_routes", description="Allows for generating paths for grid-like networks in SUMO.")
    parser.add_argument("network_file", type=str, help="Network file to generate the routes for.")
    parser.add_argument("--reproduce", action="store_true", help="Whether to run a reproduce version of the script.")
    parser.add_argument("--num-steps", type=int, default=14000, help="Number of steps all routes are generated for in each of the .rou.xml files.")
    parser.add_argument("--delta-step", type=int, default=2, help="Difference in steps between starting time of two consecutive vehicles. Halved during peak hours.")
    parser.add_argument("--peak-step-start", type=int, default=0, help="Peak time start. By default there is no peak time. During peak hours --delta-step is halved.")
    parser.add_argument("--peak-step-end", type=int, default=0, help="Peak time end. By default there is no peak time. During peak hours --delta-step is halved.")
    parser.add_argument("--bootstrap", type=int, default=1, help="Number of .rou.xml files to generate")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix with which to save .rou.xml files. By default it defaults to network file name.")
    parser.add_argument("--rand-path-prob", type=float, default=0.4, help="Percentage of random paths. Remainder is created using shortest paths.")
    parser.add_argument("--rand-path-len", type=int, default=6, help="Maximum length of a random path.")
    args = parser.parse_args()

    kwargs = vars(args)
    reproduce = kwargs.pop("reproduce")
    if reproduce:
        generate_reproduce_rou_xml_files(**kwargs)
    else:
        generate_rou_xml_files(**kwargs)
