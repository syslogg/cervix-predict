# TCC

## Cronograma:
- Estudo e desenvolvimento de uma aplicação de CNN


## 1.Dataset
Será utilizado banco de imagem com dataset convertido 

## 2. Sumário proposto

1. Introdução
2. Câncer Cervical e HPV
3. Diagnosticar o câncer cervical
    1. Exame papanicolau
    2. Diagnóstico microscópico do tecido cervical
4. Métodos de detecção de câncer cervical utilizando Aprendizagem de máquina
    1. Identificação dos padrões através de algoritmos de aprendizado
    2. Técnicas de visão computacional
5. Processamento das imagens colposcópica
    1. Filtros para criação do modelo de aprendizagem
    2. Explicar padrões criado após passar o filtro
6. Classificação das imagens utilizando redes neurais artificiais
    1. Motivação para utilização de redes neurais convolucionais.
    2. Redes Neurais Convolucionais
7. Algoritmos para classificação dos tipos do câncer
    1. Criação dos modelos de aprendizado com redes neurais convolucionais.
8. Conclusão
9. Referências Bibliográficas
10. Anexos

## 3. Estudos Sobre o Keras
### Parametros sobre a função ``Conv2D``
- **filters**: Quantidade de kernels que será aprendido naquela camada de convolução
- **kernel_size**: Tamanho da matriz do kernel (Para CONV2D, tem que ser representado por uma tupla)
- **strides**: Tupla de valores que vai ter de espaçamento para a ``feature map``. Default (1,1)
- **padding**: Aceita valores de ``valid`` ou ``same``. ``valid`` caso queira reduzir a dimensões da entrada. ``same`` para manter as mesmas dimensões da entrada.
- **data_format**: Aceita valores ``channels_last`` que os canais das imagens estão por ultimo no dataset e ``channels_first`` que os canais vem primeiro no dataset.


## Links

- https://keras.io/examples/conv_filter_visualization/
- https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd
- https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/

- [Problema de dataset com matrizes diferentes](https://datascience.stackexchange.com/questions/40462/how-to-prepare-the-varied-size-input-in-cnn-prediction)

- [Pegar imagens do arquivos para as pastas](https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720)

- [Para um dataset com tamanhos de imagens variaveis tem quqe usar uma camada de SSPNet. Uso SSPNet com Keras.](https://github.com/yhenon/keras-spp)

- [Basic of Convolution](https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/)

- https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/

- https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/

- [Explicação de Ativações: Tanh, Relu e Sigmoid](https://towardsdatascience.com/exploring-activation-functions-for-neural-networks-73498da59b02)

- [Basico sobre Keras](https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5)

- [Por que não se deve usar k-fold com CNN](https://datascience.stackexchange.com/questions/47797/using-cross-validation-technique-for-a-cnn-model/47799)

- [Possivel Solucao usando k-fold](https://datascience.stackexchange.com/questions/37009/k-fold-cross-validation-when-using-fit-generator-and-flow-from-directory-in-ke?rq=1)

- [Utilizar o ImageGenerator](https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/)