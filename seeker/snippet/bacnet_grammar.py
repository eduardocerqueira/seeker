#date: 2024-07-17T16:57:43Z
#url: https://api.github.com/gists/1dd04ba34191cc1495ebd2cc33c580ff
#owner: https://api.github.com/users/stevedh


import lrparsing
from lrparsing import Keyword, List, Prio, Ref, THIS, Token, Tokens, Repeat

class BacnetParser(lrparsing.Grammar):

 "**********"  "**********"  "**********"  "**********"  "**********"c "**********"l "**********"a "**********"s "**********"s "**********"  "**********"T "**********"( "**********"l "**********"r "**********"p "**********"a "**********"r "**********"s "**********"i "**********"n "**********"g "**********". "**********"T "**********"o "**********"k "**********"e "**********"n "**********"R "**********"e "**********"g "**********"i "**********"s "**********"t "**********"r "**********"y "**********") "**********": "**********"
        integer = "**********"="\-?[0-9]+")
        decimal = "**********"="-?[0-9]+\.[0-9]+")
        ident = "**********"="[a-zA-Z][a-zA-Z0-9\-\.\&\_]*")
        context_tag = "**********"="\[[0-9]+\]")

    definition = Ref("definition")
    optional_bacnet_type = Ref("optional_bacnet_type")
    bacnet_type = Ref("bacnet_type")

    struct_element = T.ident + optional_bacnet_type | \
        T.ident + T.context_tag + optional_bacnet_type
    enum_element = T.ident + "(" + T.integer + ")" | "..."
    integer_range = "(" + T.integer + ".." + T.integer + ")"
    decimal_range = "(" + T.decimal + ".." + T.decimal + ")"
    primative = Keyword("NULL") | Keyword("BOOLEAN") | Keyword("REAL") | \
        Keyword("Unsigned") | Keyword("Unsigned8") | \
        Keyword("Unsigned") + integer_range | \
        Keyword("Unsigned16") | Keyword("Unsigned32") | \
        Keyword("Unsigned64") | Keyword("INTEGER") | \
        Keyword("INTEGER16") | Keyword("REAL") | \
        Keyword("REAL") + decimal_range | \
        Keyword("Double") | Keyword("OCTET") + Keyword("STRING") | \
        Keyword("CharacterString") | \
        Keyword("CharacterString") + "(" + Keyword("SIZE") + integer_range + ")" | \
        Keyword("BIT") + Keyword("STRING") | \
        Keyword("ENUMERATED") | Keyword("Date") | \
        Keyword("Time") | Keyword("BACnetObjectIdentifier") 
    
    choice = Keyword("CHOICE") + "{" + List(struct_element, ",") + "}"
    sequence = Keyword("SEQUENCE") + "{" + List(struct_element, ",") + "}"
    sequence_of = Keyword("SEQUENCE") + Keyword("OF") + bacnet_type | \
        Keyword("SEQUENCE") + Keyword("SIZE") + "(" + T.integer + ")" + Keyword("OF") + bacnet_type
    enumeration = Keyword("ENUMERATED") + "{" + List(enum_element, ",") + "}"
    bit_string = Keyword("BIT") + Keyword("STRING") + "{" + List(enum_element, ",") + "}"

    bacnet_type = T.ident | primative | choice | sequence | \
                          sequence_of | enumeration | bit_string
    optional_bacnet_type = bacnet_type | bacnet_type + Keyword("OPTIONAL")

    definition = T.ident + "::=" + bacnet_type
    
    START= Repeat(definition)
    COMMENTS = Token(re="\-\-(?: "**********"
    

if __name__ == "__main__":
    import sys
    parse_tree = BacnetParser.parse(open(sys.argv[1], "r").read())
    print(BacnetParser.repr_parse_tree(parse_tree))
