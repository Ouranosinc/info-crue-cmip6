import os
print(os.environ.get('VIRTUAL_ENV'))
import xscen as xs
print(xs.__version__)

if __name__ == '__main__':
    f = open(snakemake.output, "a")
    f.write("test!")
    f.close()