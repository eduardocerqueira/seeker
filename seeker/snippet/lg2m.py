#date: 2026-03-16T17:55:05Z
#url: https://api.github.com/gists/465cf098deac246123dabb08a49b1404
#owner: https://api.github.com/users/nathanstpierre-wealthbox

#!/usr/bin/env python3
import random

L = ["large", "lazy", "little", "loud", "lonely", "lucky", "lumpy", "lanky", "lost", "livid",
     "legendary", "limping", "local", "loyal", "lurking", "lawless", "lethargic", "liquid",
     "let's", "long", "last", "lovely", "laughing", "lunatic", "liberal", "literal",
     "ludicrous", "luminous", "lopsided", "luxurious", "low-key", "lightweight", "limitless",
     "limp", "lively", "loopy", "lukewarm", "lavish", "leafy", "lean", "left-handed",
     "legitimate", "level", "licensed", "lilac", "linear", "literate", "loaded", "logical",
     "lonesome", "looming", "loose", "lopsided", "lovable", "low-budget", "lucid", "lush",
     "lyrical", "lactose-free", "laminated", "landfill", "laser-guided", "laughable",
     "law-abiding", "leading", "leaky", "leather-clad", "legally-binding", "lemony",
     "less-than-ideal", "leveraged", "life-sized", "lime-green", "llama-powered"]

G = ["gophers", "goats", "giants", "ghosts", "gnomes", "goblins", "gorillas", "guitars",
     "grandmas", "grapes", "go", "get", "grab", "gather", "generate", "grumpy", "golden",
     "galactic", "gentle", "groovy", "guys", "girls", "geese", "gifted", "gods",
     "gladiators", "glaciers", "gliders", "graduates", "gremlins", "griffins", "grocers",
     "guardians", "guerrillas", "gummies", "gurus", "gymnasts", "gazelles", "generals",
     "geniuses", "gentlemen", "giraffes", "gladly", "gleefully", "gloriously", "goalkeepers",
     "gobble", "gossip", "govern", "gracefully", "grandpas", "grasshoppers", "gravitate",
     "grenade", "grizzlies", "grooving", "groundhogs", "growing", "guarantee", "gulls",
     "gunslingers", "guppies", "gusty", "gutsy", "gamers", "gangsters", "gardeners",
     "gargoyles", "gastropods", "gatekeepers", "gazpacho-loving", "gear-shifting"]

T = ["take", "throw", "taste", "tickle", "track", "to", "the", "their", "twelve", "tiny",
     "terrible", "tremendous", "tropical", "turbo", "tangy", "tactical", "tasty", "thick",
     "through", "toward", "toasted", "thundering", "ten", "twenty", "totally",
     "thrash", "throttle", "torpedo", "transport", "traverse", "treasure", "troubleshoot",
     "tumble", "tackle", "tailgate", "tamper", "terraform", "terrorize", "theorize",
     "three", "thrifty", "tidal", "timber", "titanium", "tolerant", "top-secret",
     "torrential", "toxic", "tranquil", "translucent", "tremendous", "tricky", "triumphant",
     "trusty", "turbulent", "twice-baked", "twisted", "two-faced", "tyrannical",
     "tape-wrapped", "tax-exempt", "tectonic", "telepathic", "temporary", "tenacious",
     "thankless", "theatrical", "therapeutic", "thermodynamic", "third-party", "thorny",
     "thought-provoking", "time-traveling", "tongue-tied", "top-heavy", "triple"]

M = ["mars", "money", "muffins", "mountains", "monkeys", "music", "mustard", "missiles",
     "mangoes", "mammals", "machines", "magic", "mayhem", "motorcycles", "mushrooms",
     "mysteries", "milkshakes", "monsters", "mansions", "macaroni", "moose", "more",
     "marshmallows", "mechanics", "mermaids", "meteorites", "microchips", "midgets",
     "millennials", "miracles", "mirrors", "mischief", "modems", "molecules", "monarchs",
     "moonshine", "mosquitoes", "moustaches", "mudslides", "mummies", "munitions",
     "mutants", "magnets", "mailboxes", "mammoths", "mandolins", "marinara", "marmalade",
     "marsupials", "masterminds", "mattresses", "meatballs", "megabytes", "melodramas",
     "merchandise", "metabolism", "microwaves", "mileage", "milestones", "minivans",
     "miscreants", "molasses", "monopolies", "mortgages", "motherships", "mullets",
     "multipliers", "mythology", "maelstroms", "magnolia", "maple-syrup", "megalomania"]

for _ in range(5):
    print(f"LGTM: {random.choice(L)} {random.choice(G)} {random.choice(T)} {random.choice(M)}")
