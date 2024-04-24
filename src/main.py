from src.carregador_imagens import carregar_imagens

def main():
    caminho_imagens = '/caminho/para/a/pasta/de/imagens'
    caminho_csv = '/caminho/para/o/arquivo.csv'
    carregar_imagens(caminho_imagens, caminho_csv)

if __name__ == '__main__':
    main()