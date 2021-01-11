import getopt
import sys

if __name__ == "__main__":
    short_options = "hc:o:"
    long_options = ["help", "config", "output"]

    full_cmd_arguments = sys.argv
    argument_list = full_cmd_arguments[1:]  # remove filename

    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        # Output error, and return with an error code
        print(str(err))
        sys.exit(2)

    for current_argument, current_value in arguments:
        if current_argument in ("-h", "--help"):
            print("Required args: \n Optional args:")
        elif current_argument in ("-o", "--output"):
            print("Outputing to (%s)" % current_value)

    exit(0)
