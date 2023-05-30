"""
    Análise de sentimentos para uma base de review de filmes
    extraídos do IMDB

    Atividade: complete o código nos locais indicados

"""
import os
from collections import Counter
from typing import Any

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tf_dataset
from pandas import Series
from tensorflow.python.data.ops.dataset_ops import DatasetV2

from params import *


def convert_to_tf_dataset(df: pd.DataFrame,
                          target_column: str) -> DatasetV2:
    """
    Gera um objeto do tipo DatasetV2 a partir de um dataframe de interesse.

    :param df: [pd.DataFrame] dataframe contendo os dados a serem tratados
    :param target_column: [str] nome da coluna que contem as labels
    :return: _df_raw: [DatasetV2] dataset no formato do tf.data.Dataset
    """

    _target: Series = df.pop(target_column)
    _df_raw: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(
        (df.values,
         _target.values)
    )
    return _df_raw


def generate_train_test_validation_datasets(df: DatasetV2,
                                            size: int,
                                            train_size: int,
                                            test_size: int,
                                            shuffle: bool = False,
                                            random_state: int = 42) -> tuple[DatasetV2, DatasetV2, DatasetV2]:
    """
    Gera os datasets de treino, teste e validação

    :param df: [DatasetV2] dataset no formato do tf.data.
    :param size: [int] número de registros no dataset de entrada
    :param train_size: [int] tamanho do dataset de treino
    :param test_size: [int] tamanho do dataset de teste
    :param shuffle: [bool] se devemos realizar um shuffle a cada iteração. Default: False
    :param random_state: [int] seed para o gerador de números aleatórios. Default: 42
    :return: tuple[DatasetV2, DatasetV2, DatasetV2] Os 3 datasets: treino, teste, valid
    """
    tf.random.set_seed(random_state)
    _df: DatasetV2 = df.shuffle(size, reshuffle_each_iteration=shuffle)
    _df_test: DatasetV2 = df.take(test_size)
    _df_train_valid: DatasetV2 = df.skip(test_size)
    _df_train: DatasetV2 = _df_train_valid.take(train_size)
    _df_valid: DatasetV2 = _df_train_valid.skip(train_size)

    return _df_test, _df_train, _df_valid


def get_tokens_from_corpora(df: DatasetV2,
                            encoder: tf_dataset.deprecated.text.Tokenizer) -> Counter[Any] | Counter[str]:
    """
    Coleta os tokens contidos no corpora

    :param encoder:
    :param df: [DatasetV2] corpora
    :return: Counter tokens cont
    """
    tokenizer = encoder
    _token_counts = Counter()
    for ex in df:
        tokens = tokenizer.tokenize(ex[0].numpy()[0])
        _token_counts.update(tokens)

    return _token_counts


def encode(text_tensor, label, encoder):
    """
    Realiza o encode do vocabulário
    :param text_tensor:
    :param label:
    :param encoder:
    :return:
    """
    _text = text_tensor.numpy()[0]
    _encoded_text = encoder.encode(_text)

    return _encoded_text, label


def encode_map_fn(text, label, encoder):
    return tf.py_function(encoder, inp=[text, label],
                          Tout=(tf.int64, tf.int64))


def main():
    """
    Função principal da aplicação
    :return: None
    """
    df = pd.read_csv(os.path.join(DATA_DIR, FILE_NAME))

    # Note que os valores da coluna target são strings.
    # Antes de continuar, converta para uma variável binária.

    # CÓDIGO AQUI

    # verificação
    data = convert_to_tf_dataset(df, TARGET_COLUMN_NAME)
    for ex in data.take(3):
        tf.print(ex[0].numpy()[0][:50], ex[1])

    train_test = generate_train_test_validation_datasets(data, sizes['full'], sizes['train'], sizes['test'])
    df_test: DatasetV2 = train_test[0]
    df_train: DatasetV2 = train_test[1]
    df_valid: DatasetV2 = train_test[2]

    # verificação
    for ex in df_train.take(3):
        tf.print(ex[0].numpy()[0][:50], ex[1])

    # Obtenção dos Tokens
    encoder = tf_dataset.deprecated.text.Tokenizer()
    vocab: Counter[Any] | Counter[str] = get_tokens_from_corpora(df_train, encoder)
    print(len(vocab))

    # Encapsulamento dos tokens do vocab em inteiros
    df_train = df_train.map(encode_map_fn)
    df_valid = df_valid.map(encode_map_fn)
    df_test = df_test.map(encode_map_fn)

    # vamos definir os batches
    train_data = df_train.padding(32, padded_shapes=([-1], []))
    valid_data = df_valid.padding(32, padded_shapes=([-1], []))
    test_data = df_test.padding(32, padded_shapes=([-1], []))

    # =======    ATIVIDADE DE CLASSE ===================

    # Afim de construi o modelo para a análise de sentimentos, precisamos de uma camada para "embedar" o nosso
    # vocabulário. O Tensoflow possui uma camada específica para isso: tf.keras.layers.Embedding(input_dim=N,
    # output_dim=M) Você deve construir uma rede sequencial com a seguinte arquitetura:
    # [Embedding()] -> [SimpleRNN()] -> [SimpleRNN()] -> [Dense(1)]

    # CÓDIGO

    # Agora vamos para a rede que irá fazer a predição de sentimentos:
    # [Embedding()] -> [Bidirectional(LSTM())] -> [Dense(64, relu)] -> [Dense(1, sigmoid)]

    # CÓDIGO

    # Os parâmetros para compilação e treino estão no params.py
    optimizer = tf.keras.optimizers.Adam(1e-3)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)


if __name__ == '__main__':
    main()
