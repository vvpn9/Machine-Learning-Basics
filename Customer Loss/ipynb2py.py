# When I am using pycharm to write the jupyter files, I found that it is hard to convert file
# into py file especially when there is a need for importting

import os

os.system('jupyter nbconvert --to script *.ipynb')

# the script will change convert all ipynb file under the same dir to py
