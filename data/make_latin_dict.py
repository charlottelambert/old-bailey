#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import sys, re

xml_path = sys.argv[1]

# Define XML tree from xmlFile
xml_tree = ET.parse(xml_path)
root = xml_tree.getroot()
latin_words = set()
for elem in root.iter("entry"):
    latin_words.add(re.sub('[0-9]', "", elem.attrib['key']).lower())

print("\n".join(latin_words))
