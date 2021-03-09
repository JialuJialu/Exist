import csv
import sys
import re
import glob

ROUND_TO_TWO = True

import os

def updated_invariant(updated_inv,fature_dict):
  gen = ''
  for k in updated_inv:
    gen = gen + k + '( '
    for key in updated_inv[k]:
      if updated_inv[k][key] != 0:
        if key in feature_dict:
          gen = gen + str(updated_inv[k][key]) + '*' +feature_dict[key] + '+ '
        else:
          gen = gen + str(updated_inv[k][key]) + '*1 + '
    gen = gen[:-2] + ') + '
  return gen[:-2]

fname = sys.argv[1]  # name of txt generated invariant

if fname[0]!='g':
  num = fname[2:fname.find('_')]
  actual_inv_path = "txt/ex" + num + '.txt'
  features_path = "feature/ex" + num + '.txt'
else:
  actual_inv_path = "txt/geo_0" 
  features_path = "feature/geo_0"+ '.txt'


gen_invariant = open("txt/"+fname, "r").read()
invariant = open(actual_inv_path,"r").read()
feature = open(features_path,"r").read() #.replace('\n\n', '\n')

features = feature.split('\n\n')
features = [features[i].split('\n') for i in range(len(features))]
type_dict = {}

for v in range(1,len(features[1])):
    type_dict[features[1][v].split(':')[0]]= features[1][v].split(':')[1]

feature_dict = {}
for v in range(1,len(features[2])):
  feature_dict[features[2][v].split(':')[0]]= features[2][v].split(':')[1]

gen_invariant = gen_invariant.split('\n\n')

nBAG = gen_invariant[2].split(', ')[2].split('=')[1]
NUM_RUNS = gen_invariant[2].split(', ')[3].split('=')[1]

gen_inv = {}
ex = 0
for i in range(4,len(gen_invariant)):
    tmp_dict = {}
    k = gen_invariant[i].split('\n')[0]
    flag = 0
    if len(gen_invariant[i].split('\n')) >= 2:
      val = gen_invariant[i].split('\n')[1].split(', ')
      for v in val:
        tmp_dict[v.split(':')[0]] = round(float(v.split(':')[1]))
        if round(float(v.split(':')[1])) > 3:
          ex = 1
        if ROUND_TO_TWO:
          if(round(float(v.split(':')[1]),2)==-0.0):
            tmp_dict[v.split(':')[0]] =0.0
            continue
          tmp_dict[v.split(':')[0]] = round(float(v.split(':')[1]),2)
        if tmp_dict[v.split(':')[0]] != 0 :
          flag=1

      if (flag == 0):
        continue
      gen_inv[k] = tmp_dict
if ex == 1:
    row = [fname, updated_invariant(gen_inv, feature_dict), "False", nBAG, NUM_RUNS]
    #print(row)
    with open('invariants/results','a') as fd:
      writer = csv.writer(fd)
      writer.writerow(row)
    break

invariant = invariant.split('\n\n')
actual_inv = {}

for i in range(4,len(invariant)):
    tmp_dict = {}
    if(invariant[i]=='\nmodel_tree_end'):
      break
    k = invariant[i].split('\n')[0]
    if len(invariant[i].split('\n')) >= 2:
      val = invariant[i].split('\n')[1].split(', ')
      for v in val:
        tmp_dict[v.split(':')[0]] = round(float(v.split(':')[1]))
        if ROUND_TO_TWO:
          tmp_dict[v.split(':')[0]] = round(float(v.split(':')[1]),2)

      actual_inv[k] = tmp_dict

updated_inv = {}
for k_ in gen_inv.keys():
    t = re.split(', ',k_)
    tmp_key = ''
    for k in t:
      if k == '':
        continue
      if type_dict[re.split('<= |> |== |!=', k)[0][1:-1]]==' bool':
        tmp = re.split('<= |> |== |!=', k)
        tmp[1]=str(int(float(tmp[1][:-1]))) + ']'
        if "<=" in k:
          tmp_key = tmp_key + '== '.join(tmp)
        else:
          tmp_key = tmp_key + '!= '.join(tmp)
      else:
        tmp_key = tmp_key + k
    updated_inv[tmp_key] = gen_inv[k_]

tmp_lis = []
key_lis = []
for key, value in updated_inv.items():
    tmp_lis.append(value)
    key_lis.append(key)

#if (updated_inv==actual_inv):
#    row = [fname,updated_invariant(updated_inv, feature_dict), updated_inv==actual_inv,nBAG,NUM_RUNS]
#    #print(row)
#    with open('invariants/results','a') as fd:
#        writer = csv.writer(fd)
#        writer.writerow(row)
#    #break

  #for i in range(len(tmp_lis)):
  #  for j in range(i,len(tmp_lis)):
  #    if tmp_lis[i] == tmp_lis[j]:
  #      #TODO : merge predicates at key_lis[i] and key_lis[j]
  #      print("TODO")

row = [fname, updated_invariant(updated_inv, feature_dict), updated_inv==actual_inv, nBAG, NUM_RUNS]
print(row)
with open('invariants/results','a') as fd:
        writer = csv.writer(fd)
        writer.writerow(row)

