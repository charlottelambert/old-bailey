#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import sys, re

# Path to XML file containing latin words
xml_path = sys.argv[1]

# Define XML tree from xmlFile
xml_tree = ET.parse(xml_path)
root = xml_tree.getroot()
# Initialize set of latin words
latin_words = set()
for elem in root.iter("entry"):
    latin_words.add(re.sub('[0-9]', "", elem.attrib['key']).lower())

# Print all found latin words
print("\n".join(latin_words))
