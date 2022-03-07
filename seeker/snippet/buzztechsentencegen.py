#date: 2022-03-07T17:00:50Z
#url: https://api.github.com/gists/5fdaaf1e5ead220ff1ef8c41296e14c2
#owner: https://api.github.com/users/IronVenom

# Generating tech buzzword filled sentences with Noam Chomsky syntactic structures
# cmd line: python buzztechsentencegen.py number_of_sentences

import random
import sys 

def assemble(*args):
    return " ".join(args)

def NP(T, N):
    return assemble(T, N)

def VP(Verb, NP):
    return assemble(Verb, NP)

def sentence(NP, VP):
    return assemble(NP, VP)

def loop(x):
    T = ["The"]
    N = ["Datafication",
        "Net Neutrality",
        "Non-Fungible Tokens (NFT)",
        "Metaverse",
        "Digital Divide",
        "Everything As A Service (Xaas)",
        "Hyper-Personalization",
        "Gamification",
        "Wantrepreneur",
        "Augmented Reality (AR)",
        "Virtual reality (VR)",
        "Robotics",
        "Smart Industry 4.0",
        "Robotic Process Automation (RPA)",
        "Neural Networks",
        "Blockchain",
        "Technological Unemployment",
        "Computer Vision",
        "Distributed Cloud",
        "Artificial Intelligence (AI)",
        "The Internet of Behaviors (IOB)",
        "Hyperautomation",
        "Extended Reality (XR)",
        "Quantum Computing",
        "5G Connectivity",
        "Voice-as-User Interface (VUI)",
        "Web3",
        "Multiexperience",
        "Apps / Web-App / Native-App / Hybrid-App",
        "Cloud-Native Platforms (CNPs)",
        "Big Data",
        "Internet of Senses",
        "Platform as a Service (PaaS)",
        "Chatbots",
        "Cloud Computing",
        "Cryptocurrency",
        "Data Mining",
        "DevOps (Development Operations)",
        "MLOps (Machine Learning Development Operations)",
        "Digitization",
        "Digital Transformation",
        "Digital Disruption",
        "Industry 4.0",
        "Internet of Things (IoT)",
        "Machine Learning (ML)",
        "Voice-as-User Interface (VUI)",
        "Actionable Analytics",
        "Microservices",
        "User Interface (UI) & User Experience (UX)",
        "Infrastructure as a Service (IaaS)"]
    Verb = ["developed", "maintained", "disrupted", "created", "mutated", "gamified", "enabled", "changed", "realized"]

    print()
    for i in range(x):
        N1, N2 = random.choice(N).lower(), random.choice(N).lower()
        T1, T2 = random.choice(T).lower(), random.choice(T).lower()
        Verb1 = random.choice(Verb).lower()
        NP1 = NP(T1, N1)
        NP2 = NP(T2, N2)
        VP1 = VP(Verb1, NP2)
        print(sentence(NP1, VP1), end=". ")
    print()

if __name__ == '__main__':
    loop(int(sys.argv[1]))