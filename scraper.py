import requests
from bs4 import BeautifulSoup

# URL e headers para a requisição
url = 'https://www.kabum.com.br/busca/placas-de-video'
headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
}

# Fazendo a requisição
site = requests.get(url, headers=headers)
soup = BeautifulSoup(site.content, 'html.parser')

# Encontrando todos os itens que correspondem ao seletor desejado
placas = soup.find_all('div', class_='sc-14d01a9f-11 jjbcdA')

# Verifica se o resultado foi encontrado antes de acessar o índice
if placas:
    # Obtendo o primeiro item da lista
    placa = placas[0]
    # Buscando o nome da placa dentro desse item
    marca = placa.find('span', class_='sc-d79c9c3f-0 nlmfp sc-14d01a9f-16 bsHaTK nameCard')
    
    if marca:
        # Exibe o texto dentro da tag <span>
        print(marca.text)
    else:
        print("Elemento 'span' com a classe especificada não encontrado.")
else:
    print("Nenhuma placa de vídeo foi encontrada com a classe especificada.")
