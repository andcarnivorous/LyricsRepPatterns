import subprocess
import re
from matplotlib import cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk.tokenize import word_tokenize
import re
from scipy import sparse

def repetitionMatrix(_input, title = "", kind = False, cmap = "Reds"):

        
        _input = _input.lower()
        _input = re.sub("[\(\)\-,;:\"\.\?\!\_\[\]]", " ", _input)
        _input = re.sub("[\n']", " ", _input)

        x = word_tokenize(_input)
        y = x

        word_freq = dict()
        set_of_x = set(x)
        for word in set_of_x:
                val = x.count(word)
                word_freq.update({word : val})
        
        all_words = []

        for i in x:
                for j in y:
                        if i == j:
                                all_words.append(word_freq.get(i))
                        else:
                                all_words.append(0)

        divider = int(len(all_words)/len(x))

        arrays = []

        for element in range(0, len(all_words), divider):
                arrays.append(np.array(all_words[element-divider:element]))

        colmap = cm.get_cmap(cmap)
        arrays = np.vstack(arrays[1:])
        sparsematrix = sparse.csr_matrix(arrays)

        if kind == "sns":
                # Plot using seaborn
                sns.heatmap(arrays, cbar = False, square = True,
                            xticklabels = 50, yticklabels = 50, cmap="binary").set_title(title)
        elif kind == "sparse":
                plt.spy(sparsematrix, markersize=3, cmap="binary")
                
        else:
#                plt.scatter(arrays[:,:], arrays[:,:], marker="s")
                plt.imshow(arrays, cmap="binary", interpolation="none")

        plt.title(title)





        

_lista = os.listdir("/path/to/lyrics/")

for x in _lista:
    with open("/path/to/lyrics/"+x, encoding="cp437") as y:
                    y= y.read()
                    
                    _list.append((y,x))


os.chdir("/path/to/img/")
print(len(_list))

for i in range(len(_list[:3])):
        try:
                plt.figure( figsize = (1,1))
                plt.axis("off")
                repetitionMatrix(_list[i][0])
                plt.savefig("%s%d_matrix.png" % (str(_list[i][1])[:-4], i))
                plt.cla()
                plt.clf()
                plt.close("all")
                print(i)
        except:
                continue

