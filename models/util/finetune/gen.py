import os, sys


with open("data/complex/data.txt","w",encoding="utf8") as f:
    for i in range(1000):
        f.write("The first value is negative. The second value is positive.[SEP]While the first value is negative, the second one is positive.\n")
        f.write("The first value is negative. The second value is negative.[SEP]Both values are negative.\n")
        f.write("The first value is positive. The second value is positive.[SEP]Fortunately, both values are positive.\n")        
        f.write("The first value is positive. The second value is negative.[SEP]Unfortunately, while the first value is positive, the second one is negative.\n")        
        f.write("A third sentence that needs corrected.[SEP]The third sentence was corrected.\n")
        f.write("The fourth sentence will be corrected.[SEP]The fourth sentence was corrected.\n")
        
with open("data/complex/eval.txt","w",encoding="utf8") as f:
    for i in range(30):
        f.write("The fourth sentence will be corrected.[SEP]The fourth sentence was corrected.\n")        
        f.write("The fifth sentence will be corrected.[SEP]The fifth sentence was corrected.\n")        
        f.write("The sixth sentence will be corrected.[SEP]The sixth sentence was corrected.\n")        