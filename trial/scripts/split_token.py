from nltk import sent_tokenize, word_tokenize
from rich import print

text = 'Barton-upon-Humber Post Office is one of 11,500 branches helping customers with access to cash\n\nThe Post Office handled a record amount of cash in July with customers either depositing or withdrawing more than \u00a33.7bn. July\'s record beat previous highs set in May, April and December. The increasing use of the Post Office to handle cash comes as the rate of closure of bank branches shows no sign of slowing. More than 6,000 have shut their doors since 2015, an average of about 50 each month.\n\nBanking hubs are slowly being opened to try to offer the public similar services. There are around 70 already in operation with 100 expected to have opened their doors by Christmas. They aim to provide access to cash for those who need it, and to allow small businesses to deposit takings. But with hubs and post offices unable to provide much more than the basic banking services, the closure of branches is likely to remain contentious for a long time to come.\n\nMairi Wingate says a lot of her customers say cash helps them budget better\n\nMairi Wingate has been the postmistress in Barton-upon-Humber for 19 years and says there are lots of reasons people prefer cash. "Well it can range from anything from people wanting \u00a310 to go to the local hairdressers to go and get their hair done ranging up to \u00a3300 or \u00a3400 and it\'s just budgeting for the weekly shopping and bills. "They know exactly what they\'ve got and they can\'t overspend what they haven\'t got." The Post Office\'s reputation has been seriously damaged in recent years because of the Horizon IT scandal which saw hundreds of postmasters and mistress wrongly convicted of stealing money. During that time however daily visits to Post Office branches have remained relatively stable at around 10 million each week, and its position on the High Street means it offers a local alternative to many people when bank branches shut down.\n\n"Cash is king" according to John, who rarely uses his cards\n\nJohn calls into the Post Office nearly every week to withdraw cash from his bank account. "\u00a350 I\'ve taken out to go shopping with, like I have done for the last 50 years. Cash is still king in my book. "I do have cards but very rarely use them. You can flash a card around without really knowing what you\'re doing," he says. "Cash in the pocket [though] you know where you are."'


print(sent_tokenize(text))


import random


def generate_sublists(strings):
    if len(strings) < 3:
        raise ValueError("The list must contain at least 3 elements.")

    sublists = []
    used_indices = set()

    while len(used_indices) < len(strings):
        start_index = random.randint(0, len(strings) - 1)

        # Ensure we don't go out of bounds
        end_index = min(start_index + random.randint(3, 5), len(strings))

        sublist = strings[start_index:end_index]
        sublists.append(sublist)

        for i in range(start_index, end_index):
            used_indices.add(i)

    return sublists


# Example usage
strings = ["a", "b", "c", "d", "e", "f", "g", "h"]
sublists = generate_sublists(strings)
print(sublists)
