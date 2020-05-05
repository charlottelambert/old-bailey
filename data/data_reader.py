#!/usr/bin/env python3

import sys, html, re, os, argparse, natsort
import xml.etree.ElementTree as ET
from tqdm import tqdm
from tei_reader import TeiReader
from bs4 import BeautifulSoup
sys.path.append("..")
from utils import *

type_dict = {
    "div1": (1, 5),
    ("div0", "sessionsPaper"): (0,4),
    ("div0", "ordinarysAccount"): (2,6)
}

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
                return "$" + base + "_unk "
        # For a general annotation, encode in format speakerType_gender
        else:
            try:
                if sub_elem.attrib['type'] == 'gender':
                    gender = sub_elem.attrib["value"]
                    if gender == "indeterminate":
                        gender = "unk"
                    added_info.append(gender)
            # If gender is not known, encode in format speakerType_unk
            except KeyError:
                return "$" + base + "_unk "


    return "$" + "_".join([base, "_".join(added_info)]) + " "


def encode_annotations(args, xml_path, txt_output_dir):
    """
        Replace parts of text file with relevant annotations (as provided on command line?)

        args: arguments from the command line
        xml_path: path to XML file

        Returns string of content of xml_path file with modified annotations (if needed)
    """
    annotations = ["persName"] # Fix this to take in as an argument
    ready_for_date = False
    skip = True
    # Define XML tree from xmlFile
    xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()
    for elem in root.iter():
        # Additional work needed to process london lives corpus
        if args.london_lives:
            # Extract date
            if elem.tag == "elementDate":
                ready_for_date = True
            elif ready_for_date and elem.tag == "date":
                # Get the date, format is DD.MM.YYYY
                date = re.split("[./]", elem.attrib["modern"].rstrip())
                # Create filename from date to make later processing easier
                id = os.path.splitext(os.path.basename(xml_path))[0]
                if args.tsv:
                    if not date[-1]: year = date[-2]
                    else: year = date[-1]

                    filename = id + "\t" + year + "\t"
                else:
                    filename = "".join(date[::-1]) + "_" + id + ".txt"

                ready_for_date = False
            # Don't include ID numbers within document
            elif elem.tag == "img":
                skip = False
                elem.text = None
            elif skip:
                elem.text = " "
                continue
            elem.text = " " if not elem.text else elem.text + " "

        # Replace relevant pieces of text with annotations if necessary
        if args.encode_annotations_general or args.encode_annotations_specific:
            if elem.tag in annotations:
                if elem.tag == "persName":
                    # Get information from subelements
                    annotated_element = encode_name(args, elem)
                else: # Implement other annotations here
                    annotated_element = ""
                # Replace info in element with new info
                elem.clear()
                elem.text = annotated_element
        if not args.london_lives and args.tsv:
            try:
            # Check if current element indicates a split between trials
                if elem.tag in type_dict:
                    range = type_dict[elem.tag]
                elif (elem.tag, elem.attrib["type"]) in type_dict:
                    range = type_dict[elem.tag, elem.attrib["type"]]
                else:
                    # Does not indicate a split between trials
                    elem.text = " " if not elem.text else elem.text + " "
                    continue
                # Identify file name and insert text
                file_name = elem.attrib["id"]
                elem.text = "SPLIT_HERE" + file_name[range[0]:range[1]] + "SPLIT_HERE" + file_name
            # Element doesn't contain the right attributes
            except KeyError:
                elem.text = " " if not elem.text else elem.text + " "
        else:
            elem.text = " " if not elem.text else elem.text + " "


    # Find root of tree, convert to string, and return
    text_from_xml = str(ET.tostring(root, encoding='ASCII', method='text'))
    # Fix issues with new lines and tabs
    # sub_str = " " if args.tsv else "\n"
    text_from_xml = html.unescape(text_from_xml).replace("\\t", " ").replace("\\n", "\n")
    if args.london_lives and args.tsv: text_from_xml = text_from_xml.replace("\\n", "\n")
    text_from_xml = re.sub("\ +", " ", text_from_xml)

    if args.london_lives:
        return text_from_xml, filename
    return text_from_xml

def split_trials(text):
    """
        Iterate over text found from file. Find indication of split between
        trials ("SPLIT_HERE\tYEAR\tID"). Return list of lines to be written to
        tsv file where each line is ID\tYEAR\tTEXT.
    """
    tsv_out = []
    lines = text.split("\n")
    text = []
    for line in lines:
        if "SPLIT_HERE" in line:
            if text:
                merged_text = re.sub("\ +", " ", " ".join(text))
                tsv_out.append(id + "\t" + year + "\t" + merged_text)
                text = []
            _, year, id = line.split("SPLIT_HERE")
            id = id.rstrip()
        elif line.strip():
            text.append(line)
    if text:
        merged_text = re.sub("\ +", " ", " ".join(text))
        tsv_out.append(id + "\t" + year + "\t" + merged_text)
    return tsv_out


def main(args):
    if not args.corpus_XML_dir:
        print("Please specify directory containing XML files")
        sys.exit(1)
    args.corpus_XML_dir = os.path.join(args.corpus_XML_dir, '')

    if args.london_lives:
        input_files = []
        for root, _, files in os.walk(args.corpus_XML_dir, topdown=False):
            input_files += [os.path.join(root, f) for f in files
                            if f[-4:] == ".xml"]
    else:
        input_files = [os.path.join(args.corpus_XML_dir, f) for f in os.listdir(args.corpus_XML_dir)
        if os.path.isfile(os.path.join(args.corpus_XML_dir, f))]

        input_files = natsort.natsorted(input_files, key=lambda x: get_order(x))

    # Define name of output directory
    base_name = os.path.dirname(args.corpus_XML_dir).rstrip("/") + "-txt"
    annotations_str = "-gen" if args.encode_annotations_general else ""
    annotations_str = "-spec" if args.encode_annotations_specific else annotations_str
    txt_output_dir = base_name + annotations_str
    if not args.tsv:
        print("Writing files to " + txt_output_dir)
        if not os.path.exists(txt_output_dir):
            os.mkdir(txt_output_dir)
    tsv_out = ["id\tyear\ttext"]
    # Go through input files and generate output files
    for i in tqdm(range(len(input_files))):
        # Get current file
        file = input_files[i]

        # Change to txt file
        if not args.london_lives and not args.tsv:
            filename = os.path.splitext(os.path.basename(file))[0] + ".txt"
            file_path = os.path.join(txt_output_dir, filename)
            if os.path.exists(file_path) and not args.overwrite:
                continue
        # Write text to txt file
        try:
            # Get string version of xml
            if args.london_lives:
                text_from_xml, filename = encode_annotations(args, file, txt_output_dir)
                text_from_xml = text_from_xml[2:-1]
                # If want to output tsv file, add to tsv list
                if args.tsv:
                    text = re.sub("\n" , " ", text_from_xml)
                    l = text_from_xml.split()
                    tsv_out.append(filename + text)
                # Otherwise, write data to file
                else:
                    file_path = os.path.join(txt_output_dir, filename)
                    if os.path.exists(file_path) and not args.overwrite:
                        continue
                    with open(file_path, "w") as txt_file:
                        txt_file.write(text_from_xml)
            else:
                text_from_xml = encode_annotations(args, file, txt_output_dir)[2:-1]
                if args.tsv:
                    tsv_out += split_trials(text_from_xml)
                # else:
                #     filename =
                #     file_path = os.path.join(txt_output_dir, filename)
                #     if os.path.exists(file_path) and not args.overwrite:
                #         continue
                #     with open(file_path, "w") as txt_file:
                #         txt_file.write(text_from_xml)

        except UnicodeDecodeError:
            print("UnicodeDecodeError reading " + file + ". Skipping...")
            continue
        except ET.ParseError:
            print("ParseError reading " + file + ". Skipping...")
    if args.tsv:
        with open(txt_output_dir + ".tsv", 'w') as f:
            # Sort in year order
            tsv_out = [tsv_out[0]] + natsort.natsorted(tsv_out[1:], key=lambda x: x.split("\t")[1])
            f.write("\n".join(tsv_out))
    print("Data written to " + txt_output_dir + ".tsv", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_XML_dir', type=str, default="/work/clambert/thesis-data/sessionsPapers", help='directory containing XML version of corpus')
    parser.add_argument('--encode_annotations_general', default=False, action="store_true", help='whether or not to encode general version of annotations in text')
    parser.add_argument('--encode_annotations_specific', default=False, action="store_true", help='whether or not to encode specific version of annotations in text')
    parser.add_argument('--overwrite', default=False, action="store_true", help='whether or not to overwrite old files with the same names')
    parser.add_argument('--london_lives', default=False, action="store_true", help='whether or not input is London Lives corpus')
    parser.add_argument('--tsv', default=1, type=int, help="whether or not to store output as tsv")
    args = parser.parse_args()
    main(args)
