#!/usr/bin/env python
# Run in Abaqus Script

from abaqus import *
from odbAccess import *
from abaqusConstants import *
import sys
import os
import csv

# Module type
module_type = sys.argv[-1]

# Set path
path = "./data/" + module_type + "/"
odb_path = os.path.join(path, "abq_file/")

# Get odb file names
odb_csv_file = open(os.path.join(path, "cpl_odb.csv"), "r")
odb_csv_reader = csv.reader(odb_csv_file)

for odb_name in odb_csv_reader:
    # Open odb file
    odb = openOdb(os.path.join(odb_path, "".join(odb_name)))

    # Get node sets
    node_set = odb.rootAssembly.nodeSets["SURFACE"]

    # Get node set values
    field = odb.steps["Step-1"].frames[-1].fieldOutputs["U"]
    node_values = field.getSubset(region=node_set).values

    node_value_list = []
    for node_value in node_values:
        node_value_list.append([node_value.nodeLabel, node_value.data])

    # # Get forces
    region = odb.steps["Step-1"].historyRegions["Surface BOTTOM_SURFACE"]
    sof1 = region.historyOutputs["SOF1  on section I-SECTION-1"].data
    sof2 = region.historyOutputs["SOF2  on section I-SECTION-1"].data
    sof3 = region.historyOutputs["SOF3  on section I-SECTION-1"].data
    som1 = region.historyOutputs["SOM1  on section I-SECTION-1"].data
    som2 = region.historyOutputs["SOM2  on section I-SECTION-1"].data
    som3 = region.historyOutputs["SOM3  on section I-SECTION-1"].data

    # Get pose
    pose = []
    pose_file = open(os.path.join(path, "pose.csv"), "r")
    pose_data = csv.reader(pose_file)
    for i, row in enumerate(pose_data):
        if i == int("".join(odb_name).replace(".odb", "")) - 1:
            pose = row
            break

    # Write csv file
    data_csv_name = "".join(odb_name).replace(".odb", ".csv")
    data_csv_file = open(os.path.join(odb_path, data_csv_name), "wb")
    data_csv_writer = csv.writer(data_csv_file)
    data_csv_writer.writerow(pose)
    data_csv_writer.writerow(
        [sof1[-1][1], sof2[-1][1], sof3[-1][1], som1[-1][1], som2[-1][1], som3[-1][1]]
    )
    for node_value in node_value_list:
        data_csv_writer.writerow(
            [node_value[0], node_value[1][0], node_value[1][1], node_value[1][2]]
        )
    data_csv_file.close()
    odb.close()
    print(data_csv_name + " done.")
odb_csv_file.close()

print("All done.")
