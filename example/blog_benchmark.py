from pgmpy.readwrite import BIFReader, BIFWriter
import os
from bif2blog import BLOGWriter
for file in os.listdir("./"):
    if file.endswith(".bif"):
        print("convert BIF file:", file)
        model = BIFReader(file).get_model()
        writer = BLOGWriter(model)
        new_name = os.path.splitext(file)[0]+".blog"
        writer.write_BLOG(filename=new_name)
