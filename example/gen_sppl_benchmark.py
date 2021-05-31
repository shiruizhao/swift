from pgmpy.readwrite import BIFReader, BIFWriter
import os
from bif2SPPL import SPPLWriter
for file in os.listdir("./"):
    if file.endswith(".bif"):
        print("convert BIF file:", file)
        model = BIFReader(file).get_model()
        writer = SPPLWriter(model)
        new_name = os.path.splitext(file)[0]+".sppl"
        writer.write_SPPL(filename=new_name)
