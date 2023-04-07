#date: 2023-04-07T16:51:02Z
#url: https://api.github.com/gists/b5b7c71d5ccf5169f437661d10ea2f3f
#owner: https://api.github.com/users/yrHeTaTeJlb

# Qt Creator debug helpers for Unreal Engine 
# https://doc.qt.io/qtcreator/creator-debugging-helpers.html#debugging-helper-implementation

def print_members(value, level = 0):
    print(level * 2 * ' ', value)
    for member in value.members(True):
        print_members(member, level + 1)

def qdump__FString(d, value):
    data_member = value["Data"]
    print(data_member.type.templateArgument(0))

    allocator_data_member = data_member["AllocatorInstance"][0][0]

    size = max(data_member["ArrayNum"].integer() - 1, 0)
    data = allocator_data_member.pointer()
    char_type = data_member.type.templateArgument(0)

    d.putCharArrayHelper(data, size, char_type, d.currentItemFormat())


def qdump__FName(d, value):
    display_name_entry = d.call("FNameEntry", value, "GetDisplayNameEntry").dereference()
    header_member = display_name_entry["Header"]
    is_wide = bool(header_member["bIsWide"].integer())
    if is_wide:
        name_member = display_name_entry["WideName"]
    else:
        name_member = display_name_entry["AnsiName"]

    size = header_member["Len"].integer()
    data = name_member.address()
    char_type = name_member.type.unqualified().ltarget.stripTypedefs()

    d.putCharArrayHelper(data, size, char_type, d.currentItemFormat())

def qdump__TArray(d, value):
    allocator_data_member = value["AllocatorInstance"][0][0]

    data = allocator_data_member.pointer()
    size = value["ArrayNum"].integer()
    item_type = value.type.templateArgument(0)

    d.putArrayData(data, size, item_type)
