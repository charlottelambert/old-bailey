#!/usr/bin/env python3

import sys
import os, argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm
from tei_reader import TeiReader

# Can't figure out how to access the individual elements
# def tei_encode_annotation(args, xml_path):
#     reader = TeiReader()
#     xml_content = reader.read_file(xml_path)
#     print(list(list(xml_content.documents)[0].attributes))
    # print(xml_content.documents)
    # print(len(xml_content.attributes))
    # for o in xml_content.divisions:
    #     print("HI" + o)

def encode_name(args, elem):
    """
        Encode name information based on a given element with tag "persName"

        args: arguments from the command line
        elem: an ElementTree.Element object

        returns information to replace name with.
    """
    base = elem.tag[:-4] if not "type" in elem.attrib else elem.attrib["type"][:-4]
    added_info = []

    # Iterate over subelements.
    for sub_elem in elem.iter():
        # For a specific annotation, encode in format speakerType_GIVEN_SURNAME
        if args.encode_annotations_specific:
            try:
                if sub_elem.attrib['type'] == 'surname':
                    surname = sub_elem.attrib["value"].split(" ")
                    added_info.append("_".join(surname))
                elif sub_elem.attrib['type'] == 'given':
                    given = sub_elem.attrib["value"].split(" ")
                    added_info = ["_".join(given)] + added_info
            # If name is not known, encode in format speakerType_unk
            except KeyError:
                return base + "_unk"
        # For a general annotation, encode in format speakerType_gender
        else:
            try:
                if sub_elem.attrib['type'] == 'gender':
                    added_info.append(sub_elem.attrib["value"])
            # If gender is not known, encode in format speakerType_unk
            except KeyError:
                return base + "_unk"

    return "_".join([base, "_".join(added_info)])


def encode_annotations(args, xml_path):
    """
        Replace parts of text file with relevant annotations (as provided on command line?)

        args: arguments from the command line
        xml_path: path to XML file

        Returns string of content of xml_path file with modified annotations (if needed)
    """
    annotations = ["persName"] # Fix this to take in as an argument

    # Define XML tree from xmlFile
    xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()

    # Replace relevant pieces of text with annotations if necessary
    if args.encode_annotations_general or args.encode_annotations_specific:
        # Go through each element in the xmlTree
        for elem in root.iter():
            if elem.tag in annotations:
                if elem.tag == "persName":
                    # Get information from subelements
                    annotated_element = encode_name(args, elem)
                else: # Implement other annotations here
                    annotated_element = ""
                # Replace info in element with new info
                elem.clear()
                elem.text = annotated_element

    # Find root of tree, convert to string, and return
    text_from_xml = str(ET.tostring(root, encoding='utf-8', method='text'))
    return text_from_xml.replace('\\n', '\n') # Fixes issue printing "\n"


def main(args):
    if not args.corpus_XML_dir:
        print("Please specify directory containing XML files")

    input_files = [os.path.join(args.corpus_XML_dir, f) for f in os.listdir(args.corpus_XML_dir) if os.path.isfile(os.path.join(args.corpus_XML_dir, f))]

    # Define name of output directory
    base_name = os.path.dirname(args.corpus_XML_dir) + "-txt"
    annotations_str = "-gen_annotations" if args.encode_annotations_general else ""
    annotations_str = "-spec_annotations" if args.encode_annotations_specific else annotations_str
    txt_output_dir = base_name + annotations_str
    print("Writing files to " + txt_output_dir)
    
    if not os.path.exists(txt_output_dir):
        os.mkdir(txt_output_dir)

    # Go through input files and generate output files
    for i in tqdm(range(len(input_files))):
        # Get current file
        file = input_files[i]

        # Change to txt file
        filename = os.path.splitext(os.path.basename(file))[0] + ".txt"
        file_path = os.path.join(txt_output_dir, filename)
        if os.path.exists(file_path) and not args.overwrite:
            continue

        # Write text to txt file
        with open(os.path.join(txt_output_dir, filename), "w+") as txt_file:
            try:
                # Get string version of xml
                text_from_xml = encode_annotations(args, file)[2:-1]
            except UnicodeDecodeError:
                print("Error reading " + file + ". Skipping...")
                continue
            except ET.ParseError:
                print("Error reading " + file + ". Skipping...")
            txt_file.write(text_from_xml)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_XML_dir', type=str, default="../data/sessionsPapers", help='directory containing XML version of corpus')
    parser.add_argument('--encode_annotations_general', default=False, action="store_true", help='whether or not to encode general version of annotations in text')
    parser.add_argument('--encode_annotations_specific', default=False, action="store_true", help='whether or not to encode specific version of annotations in text')
    parser.add_argument('--overwrite', default=False, action="store_true", help='whether or not to overwrite old files with the same names')
    args = parser.parse_args()
    main(args)
