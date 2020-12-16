import traci
import argparse
from pathlib import Path


def generate_e2_detectors(network_file: str, length: int, out_file: str):
    if out_file is None:
        # Assuming .net.xml
        out_file_name = Path(network_file).name[:-8] + ".add.xml"
        out_file = str(Path(network_file).parent / out_file_name)
    xml_lines = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
        "<additional xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"http://sumo.dlr.de/xsd/additional_file.xsd\">\n",
    ]

    traci.start(["sumo", "-n", network_file])
    for ts_id in traci.trafficlight.getIDList():
        incoming_lanes = set([link[0][0] for link in traci.trafficlight.getControlledLinks(ts_id)])
        for l_id in incoming_lanes:
            l_len = traci.lane.getLength(l_id)
            used_len = min(l_len, length)
            xml_lines.append(
                f"\t<laneAreaDetector id=\"e2_{l_id}\" lane=\"{l_id}\" length=\"{used_len:.2f}\" endPos=\"{l_len:.2f}\" tl=\"{ts_id}\" file=\"additionals/e2_{l_id}.log\"/>\n"
            )
    xml_lines.append("</additional>\n")
    traci.close()

    with open(out_file, mode="w") as f:
        f.writelines(xml_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "generate_routes",
        description="Allows for generating additional e2 detectors for incoming lanes."
    )
    parser.add_argument("network_file", type=str, help="Network file to generate the detectors for.")
    parser.add_argument("--length", type=int, default=150, help="Length of the detectors")
    parser.add_argument(
        "--out-file",
        type=str,
        default=None,
        help="File to save the additionals to. Should end with .add.xml",
    )
    args = parser.parse_args()
    kwargs = vars(args)
    generate_e2_detectors(**kwargs)
