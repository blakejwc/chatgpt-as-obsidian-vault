import json
import argparse
from jsonschema import FormatChecker, validate, exceptions


class QuietFormatChecker(FormatChecker):
    def check_format(self, format, value):
        try:
            super().check_format(format, value)
        except exceptions.FormatError:
            pass


def main(schema_file, data_file, quiet):
    # Load the schema
    with open(schema_file) as f:
        schema = json.load(f)

    # Load the JSON data
    with open(data_file) as f:
        data = json.load(f)

    # Validate the data
    try:
        if(quiet):
            validate(instance=data, schema=schema, format_checker=QuietFormatChecker())
        else:
            validate(instance=data, schema=schema)
        print("Validation successful!")
    except exceptions.ValidationError as e:
        print("Validation error:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate JSON data against a JSON Schema.')
    parser.add_argument('schema_file', type=str, help='Path to the JSON Schema file.')
    parser.add_argument('data_file', type=str, help='Path to the JSON data file.')
    parser.add_argument('--quiet', action='store_true', help='Suppress format validation warnings.')
    args = parser.parse_args()

    print(args)
    main(args.schema_file, args.data_file, args.quiet)