import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.generators import TextGenerator
from src.utils import analyze_pos, load_and_preprocess_data


def main():
    parser = argparse.ArgumentParser(description='Generate text using LSTM')
    parser.add_argument('seed_text', type=str, help='Seed text for generation')
    args = parser.parse_args()

    documents = load_and_preprocess_data()
    generator = TextGenerator()
    generator.train_model(documents)

    generated_text = generator.generate(args.seed_text, num_words=25)

    print(f'\nGenerated text:\n{generated_text}\n')

    analysis = analyze_pos(generated_text)

    print('POS Analysis:')
    print(f'Nouns: {", ".join(analysis['nouns'])}')
    print(f'Verbs: {", ".join(analysis['verbs'])}')
    print(f'Adjectives: {", ".join(analysis['adjectives'])}')


if __name__ == '__main__':
    main()
