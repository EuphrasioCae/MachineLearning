import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def limpar_terminal():
    os.system('cls')

"""
    DESAFIO:    
    
    Desenvolver um script em Python que seja capaz de classificar textos curtos
    em categorias pré-definidas pelo usuário (por exemplo: "tecnologia", "esporte",
    ou "política"). Tem o objetivo de treinar um modelo simples de Machine Learning.

    Comando para inicializar o script no terminal do VsCode: python indentifyText.py
"""
  
# Dados de Treinamento
textos = [

# tecnologia
"O novo lançamento da Apple",
"Atualização no mundo da tecnologia",
"O novo lançamento da Samsung",
"Empresa anuncia novo smartphone",
"Atualização de software para celulares",
"Novo processador para computadores foi lançado",
"Gigante da tecnologia apresenta novo produto",
"Sistema operacional recebe nova atualização",
"Empresa de tecnologia investe em inteligência artificial",
"Novo notebook chega ao mercado",
"Startup desenvolve aplicativo inovador",
"Atualização de segurança para dispositivos móveis",
"Empresa anuncia novo chip para computadores",
"Lançamento de novo smartwatch",
"Tecnologia 5G avança no país",
"Fabricante apresenta novo telefone",
"Empresa revela novo tablet",
"Lançamento de novo dispositivo eletrônico",
"Atualização do sistema Android disponível",
"Nova versão de aplicativo é lançada",
"Empresa lança novo headset",
"Fabricante anuncia novo monitor gamer",
"Novo hardware melhora desempenho do computador",
"Empresa investe em computação em nuvem",
"Nova plataforma digital é apresentada",
"Empresa lança nova tecnologia de bateria",
"Novo sensor tecnológico é desenvolvido",
"Atualização de firmware para dispositivos",
"Empresa anuncia nova geração de processadores",
"Lançamento de smartphone com câmera avançada",
"Empresa lança nova linha de smartphones",
"Novo sistema de inteligência artificial é apresentado",
"Atualização melhora desempenho do computador",
"Empresa apresenta nova tecnologia de câmera",
"Lançamento de novo tablet no mercado",
"Fabricante anuncia novo processador gráfico",
"Atualização do aplicativo melhora segurança",
"Empresa investe em tecnologia de realidade virtual",
"Novo gadget tecnológico é apresentado",
"Empresa lança nova versão do sistema operacional",
"Lançamento de tecnologia para carros autônomos",
"Empresa desenvolve novo software empresarial",
"Nova geração de notebooks é anunciada",
"Atualização melhora performance do celular",
"Empresa anuncia novo dispositivo wearable",
"Startup apresenta solução tecnológica inovadora",
"Lançamento de novo dispositivo inteligente",
"Empresa lança nova plataforma digital",
"Atualização tecnológica melhora conectividade",
"Fabricante revela novo modelo de smartphone",

# esportes
"Resultado do jogo de ontem",
"Campeonato de futebol",
"Resultado da partida de ontem",
"Time vence a final do campeonato",
"Jogador marca três gols na partida",
"Equipe conquista título nacional",
"Partida terminou empatada",
"Treinador comenta estratégia do time",
"Clube anuncia contratação de jogador",
"Final do torneio acontece amanhã",
"Seleção vence jogo amistoso",
"Equipe se prepara para o campeonato",
"Atleta bate recorde na competição",
"Torcedores comemoram vitória do time",
"Clube vence partida decisiva",
"Jogador sofre lesão durante partida",
"Equipe anuncia novo técnico",
"Time se classifica para semifinal",
"Partida termina com vitória do visitante",
"Equipe vence campeonato regional",
"Atleta conquista medalha de ouro",
"Time inicia preparação para torneio",
"Jogador é eleito melhor da partida",
"Clube apresenta novo uniforme",
"Competição internacional começa hoje",
"Equipe disputa final do campeonato",
"Time marca gol nos minutos finais",
"Treinador analisa desempenho do time",
"Seleção nacional convoca novos jogadores",
"Clube vence clássico regional",
"Time conquista vitória importante no campeonato",
"Jogador marca gol decisivo na partida",
"Equipe vence jogo emocionante",
"Clube anuncia reforço para a temporada",
"Time inicia preparação para final do torneio",
"Atleta conquista medalha em competição internacional",
"Treinador comenta desempenho da equipe",
"Clube vence partida fora de casa",
"Seleção se prepara para campeonato mundial",
"Equipe garante vaga na final",
"Jogador é destaque da rodada",
"Clube apresenta novo técnico",
"Time vence clássico do campeonato",
"Equipe disputa semifinal do torneio",
"Atleta quebra recorde nacional",
"Partida termina com vitória do mandante",
"Seleção nacional convoca novos atletas",
"Equipe vence campeonato regional",
"Jogador se destaca na competição",
"Clube comemora vitória importante",

# política
"Eleições presidenciais",
"Política internacional",
"Atualização sobre a geopolítica",
"Debate entre candidatos à presidência",
"Congresso aprova nova lei",
"Governo anuncia nova política econômica",
"Discussão política no parlamento",
"Presidente faz pronunciamento oficial",
"Senado debate reforma tributária",
"Ministro anuncia novas medidas do governo",
"País discute relações diplomáticas",
"Câmara vota projeto de lei",
"Governador anuncia novo programa social",
"Partidos políticos discutem alianças",
"Debate sobre políticas públicas",
"Lançamento de nova lei pelo governo",
"Lançamento de programa social",
"Governo anuncia lançamento de nova política",
"Lançamento de reforma política",
"Supremo tribunal analisa nova lei",
"Congresso discute reforma administrativa",
"Ministro participa de reunião diplomática",
"Presidente anuncia nova medida econômica",
"Debate sobre orçamento público",
"Senado aprova projeto de lei",
"Governador propõe nova política pública",
"Partido político apresenta proposta de reforma",
"Governo federal anuncia nova estratégia econômica",
"Deputados discutem nova legislação",
"Reunião política define novas diretrizes",
"Governo apresenta nova proposta de lei",
"Congresso discute reforma política",
"Presidente anuncia nova medida econômica",
"Senado aprova projeto de lei",
"Deputados debatem nova legislação",
"Governo lança programa social",
"Reunião política define novas estratégias",
"Ministro anuncia política pública",
"Congresso vota nova proposta econômica",
"Debate político ocorre no parlamento",
"Presidente participa de reunião diplomática",
"Senadores discutem nova reforma",
"Governo propõe nova política educacional",
"Partidos discutem alianças eleitorais",
"Congresso aprova medida provisória",
"Governador anuncia nova política estadual",
"Debate sobre economia ocorre no congresso",
"Nova proposta de reforma é apresentada",
"Autoridades discutem política internacional",
"Governo anuncia nova legislação"
]

categorias = [

# tecnologia
"tecnologia","tecnologia","tecnologia","tecnologia","tecnologia",
"tecnologia","tecnologia","tecnologia","tecnologia","tecnologia",
"tecnologia","tecnologia","tecnologia","tecnologia","tecnologia",
"tecnologia","tecnologia","tecnologia","tecnologia","tecnologia",
"tecnologia","tecnologia","tecnologia","tecnologia","tecnologia",
"tecnologia","tecnologia","tecnologia","tecnologia","tecnologia",
"tecnologia","tecnologia","tecnologia","tecnologia","tecnologia",
"tecnologia","tecnologia","tecnologia","tecnologia","tecnologia",
"tecnologia","tecnologia","tecnologia","tecnologia","tecnologia",
"tecnologia","tecnologia","tecnologia","tecnologia","tecnologia",

# esportes
"esportes","esportes","esportes","esportes","esportes",
"esportes","esportes","esportes","esportes","esportes",
"esportes","esportes","esportes","esportes","esportes",
"esportes","esportes","esportes","esportes","esportes",
"esportes","esportes","esportes","esportes","esportes",
"esportes","esportes","esportes","esportes","esportes",
"esportes","esportes","esportes","esportes","esportes",
"esportes","esportes","esportes","esportes","esportes",
"esportes","esportes","esportes","esportes","esportes",
"esportes","esportes","esportes","esportes","esportes",

# política
"política","política","política","política","política",
"política","política","política","política","política",
"política","política","política","política","política",
"política","política","política","política","política",
"política","política","política","política","política",
"política","política","política","política","política",
"política","política","política","política","política",
"política","política","política","política","política",
"política","política","política","política","política",
"política","política","política","política","política"
]
  
  # Convertendo textos em uma matriz de contagens de tokens
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words=[
        "de","da","do","das","dos",
        "para","uma","um","o","a","no","na"
    ]
)
X = vectorizer.fit_transform(textos)
  
  # Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    categorias,
    test_size=0.2,
    random_state=42,
    stratify=categorias
)
  
  # Treinando o classificador
clf = LinearSVC()
clf.fit(X_train, y_train)
  
  # Predição e Avaliação
y_pred = clf.predict(X_test)
limpar_terminal()
print(f"Acurácia: {accuracy_score(y_test, y_pred)}")

# =================================
# Classificação de texto digitado
# =================================
while True:
    texto_usuario = input("\nDigite um texto (ou 'sair'): ")

    if texto_usuario.lower() == "sair":
        break

    vetor = vectorizer.transform([texto_usuario])
    categoria = clf.predict(vetor)

    print("Categoria:", categoria[0])