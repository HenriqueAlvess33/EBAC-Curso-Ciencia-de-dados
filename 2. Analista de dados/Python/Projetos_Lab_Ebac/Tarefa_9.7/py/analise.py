import os
import json
import csv
from datetime import datetime
from sys import argv

import requests
import seaborn as sns

URL = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.4392/dados'

data_e_hora = datetime.now()
data = datetime.strftime(data_e_hora, '%Y/%m/%d')
hora = datetime.strftime(data_e_hora, '%H:%M:%S')

try:
    response = requests.get(URL)
    response.raise_for_status()
except requests.HTTPError as exc:
    print('Dado não encontrado, continuando.')
    cdi = None
except Exception as exc:
    print('Erro, parando a execução')
    raise exc
else:
    dado = json.loads(response.text)[-1] ['valor']
    cdi = float(dado)
    

if os.path.exists == False:
    with open(file = './taxa-cdi.csv', mode = 'w', encoding = 'utf8') as fp:
        fp.write('data,hora,taxa\n')
        
with open(file = './taxa-cdi.csv', mode = 'a', encoding = 'utf8') as fp:
    fp.write(f'{data}, {hora}, {cdi}\n')
    
print('Sucesso')


horas = []
taxas = []

with open(file='./taxa-cdi.csv', mode='r', encoding='utf8') as fp:
  linha = fp.readline()
  linha = fp.readline()
  while linha:
    linha_separada = linha.split(sep=',')
    hora = linha_separada[1]
    horas.append(hora)
    taxa = float(linha_separada[2])
    taxas.append(taxa)
    linha = fp.readline()

# Salvando no grafico

grafico = sns.lineplot(x=horas, y=taxas)
grafico.get_figure().savefig(f"{argv[1]}.png")