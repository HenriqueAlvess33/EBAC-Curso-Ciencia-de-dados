import os
import json
from datetime import datetime

import requests

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
