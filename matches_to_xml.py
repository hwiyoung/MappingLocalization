# from xml.etree.ElementTree import Element, dump
import xml.etree.ElementTree as ET

# def sensors_xml(node1):
#     node1_1 = Element("sensor", id="0", label="train_L", type="frame")
#     node1.append(node1_1)
#
#     node1_1_1 = Element("resolution", width="1384", height="1032")
#     node1_1.append(node1_1_1)
#     node1_1_2 = Element("property", name="pixel_width", value="0.00645")
#     node1_1.append(node1_1_2)
#     node1_1_3 = Element("property", name="pixel_height", value="0.00645")
#     node1_1.append(node1_1_3)
#     node1_1_4 = Element("property", name="focal_length", value="8.25")
#     node1_1.append(node1_1_4)
#     node1_1_5 = Element("property", name="fixed", value="true")
#     node1_1.append(node1_1_5)
#     node1_1_6 = Element("property", name="layer_index", value="0")
#     node1_1.append(node1_1_6)
#     node1_1_7 = Element("bands")
#     node1_1.append(node1_1_7)
#
#     node1_1_7_1 = Element("band", label="Red")
#     node1_1_7.append(node1_1_7_1)
#     node1_1_7_2 = Element("band", label="Green")
#     node1_1_7.append(node1_1_7_2)
#     node1_1_7_3 = Element("band", label="Blue")
#     node1_1_7.append(node1_1_7_3)
#
#     node1_1_8 = Element("bands")
#     node1_1_8.text = "uint8"
#     node1_1.append(node1_1_8)
#
#     node1_1_9 = Element("calibration", frame="frame")  # , class="initial")
#     node1_1.append(node1_1_9)
#
#     node1_1_9_1 = Element("resolution", width="1384", height="1032")
#     node1_1_9.append(node1_1_9_1)
#     node1_1_9_2 = Element("f")
#     node1_1_9_2.text = "1279.71039201197"
#     node1_1_9.append(node1_1_9_2)
#     node1_1_9_2 = Element("cx")
#     node1_1_9_2.text = "-6.272717931"
#     node1_1_9.append(node1_1_9_2)
#
#
# def cameras_xml(node2, eo):
#     next_id = len(eo)
#     for i in range(next_id):
#         node2_1 = Element("camera", id=str(i), sensor_id="수정", label=eo[i, 0])
#         node2.append(node2_1)
#
#
# def write_to_xml(eo):
#     root = Element("document", version="1.5.0")
#
#     node = Element("chunk", label="Merged Chunk", enabled="true")
#     root.append(node)
#
#     node1 = Element("sensors", next_id="4")
#     node.append(node1)
#
#     sensors_xml(node1)
#     sensors_xml(node1)
#
#     node2 = Element("cameras", next_id="수정", next_group_id="0")
#     node.append(node2)
#
#     cameras_xml(node2, eo)
#
#     # node3 = Element("markers", next_id="12", next_group_id="0")
#     node3 = Element("markers", next_id="수정", next_group_id="0")
#     node.append(node3)
#
#     node4 = Element("reference")
#     node4.text = "LOCAL_CS[\"Local Coordinates (m)\",LOCAL_DATUM[\"Local Datum\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]]]"
#     node.append(node4)
#
#     node5 = Element("settings")
#     node.append(node5)
#
#     node6 = Element("meta")
#     node.append(node6)
#
#     node7 = Element("frames", next_id="1")
#     node.append(node7)
#
#     node2_1 = Element("camera", id="수정", sensor_id="수정", label="수정")
#     node2.append(node2_1)
#
#     node2_1_1 = Element("reference", x="수정", y="수정", z="수정", yaw="수정", pitch="수정", roll="수정", enabled="true")
#     node2_1.append(node2_1_1)
#
#     node5_1 = Element("property", name="accuracy_tiepoints", value="1")
#     node5.append(node5_1)
#
#     node6_1 = Element("property", name="AlignCameras/adaptive_fitting", value="false")
#     node6.append(node6_1)
#
#     node7_1 = Element("frame", id="0")
#     node7.append(node7_1)
#
#     node7_1_1 = Element("markers")
#     node7_1.append(node7_1_1)
#
#     node7_1_1_1 = Element("marker", marker_id="수정")
#     node7_1_1.append(node7_1_1_1)
#
#     node7_1_1_1_1 = Element("location", camera_id="수정", pinned="true", x="수정", y="수정")
#     node7_1_1_1.append(node7_1_1_1_1)
#
#     return root

interval = 10
def write_to_xml(xml, matches, seq):
    # parse xml file
    doc = ET.parse(xml)

    # get root node
    root = doc.getroot()

    chunk = root.findall("chunk")

    # markers
    node1 = ET.Element("markers", next_id=str(len(matches)), next_group_id="0")
    chunk[0].append(node1)

    for i in range(len(matches)):
        node1_1 = ET.Element("marker", id=str(i), label=' '.join(["point", str(i)]))
        node1.append(node1_1)

    # frames
    node2 = ET.Element("frames", next_id="1")
    chunk[0].append(node2)

    node2_1 = ET.Element("frame", id="0")
    node2.append(node2_1)

    node2_1_1 = ET.Element("markers")
    node2_1.append(node2_1_1)

    center_L_of_train = int((seq*2*2)/2)
    # test(49_L)|train|train-20|train-10|train+10|train+20
    cam_idx = [109, center_L_of_train, center_L_of_train - interval*2*2, center_L_of_train - interval*2,
               center_L_of_train + interval*2, center_L_of_train + interval*2*2]
    # # test|train|train-20|train+10
    # cam_idx = [109, center_L_of_train, center_L_of_train - interval*2*2, center_L_of_train + interval*2]
    for i in range(len(matches)):
        node2_1_1_1 = ET.Element("marker", marker_id=str(i))
        node2_1_1.append(node2_1_1_1)
        for j in range(int(matches.shape[1]/2)):
            if matches[i, 2*j] == -1 or matches[i, 2*j+1] == -1:
                pass
            else:
                node2_1_1_1_1 = ET.Element("location", camera_id=str(cam_idx[j]), pinned="true",
                                           x=str(matches[i, 2*j]), y=str(matches[i, 2*j+1]))
                node2_1_1_1.append(node2_1_1_1_1)

    indent(root)
    ET.dump(root)

    # https://stackoverflow.com/questions/3605680/creating-a-simple-xml-file-using-python
    tree = ET.ElementTree(root)
    ET.dump(tree)
    tree.write('markers.xml')


def write_to_xml2(xml, matches, seq):
    # parse xml file
    doc = ET.parse(xml)

    # get root node
    root = doc.getroot()

    chunk = root.findall("chunk")

    # markers
    node1 = ET.Element("markers", next_id=str(len(matches)), next_group_id="0")
    chunk[0].append(node1)

    for i in range(len(matches)):
        node1_1 = ET.Element("marker", id=str(i), label=' '.join(["point", str(i)]))
        node1.append(node1_1)

    # frames
    node2 = ET.Element("frames", next_id="1")
    chunk[0].append(node2)

    node2_1 = ET.Element("frame", id="0")
    node2.append(node2_1)

    node2_1_1 = ET.Element("markers")
    node2_1.append(node2_1_1)

    center_L_of_train = int((seq*2*2)/2)
    # test(49_L)|train|train-20|train-10|train+10|train+20
    cam_idx = [109, center_L_of_train, center_L_of_train - interval*2*2, center_L_of_train - interval*2,
               center_L_of_train + interval*2, center_L_of_train + interval*2*2]
    # # test|train|train-20|train+10
    # cam_idx = [109, center_L_of_train, center_L_of_train - interval*2*2, center_L_of_train + interval*2]
    for i in range(len(matches)):
        node2_1_1_1 = ET.Element("marker", marker_id=str(i))
        node2_1_1.append(node2_1_1_1)
        for j in range(int(matches.shape[1]/2)):
            if matches[i, 2*j] == -1 or matches[i, 2*j+1] == -1:
                pass
            else:
                node2_1_1_1_1 = ET.Element("location", camera_id=str(cam_idx[j]), pinned="true",
                                           x=str(matches[i, 2*j]), y=str(matches[i, 2*j+1]))
                node2_1_1_1.append(node2_1_1_1_1)

    indent(root)
    ET.dump(root)

    # https://stackoverflow.com/questions/3605680/creating-a-simple-xml-file-using-python
    tree = ET.ElementTree(root)
    ET.dump(tree)
    tree.write('markers.xml')

def indent(elem, level=0): #자료 출처 https://goo.gl/J8VoDK
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# root = write_to_xml()
# indent(root)
# dump(root)
