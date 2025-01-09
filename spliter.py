def text_spliter(text: str, chunk_size: int, cover_size: int):
    """
    文本分割
    param: text: str 文本
    param: chunk_size int 切分块的大小
    param: cover_size int 前后重复覆盖块的大小
    return
    """

    sentences = []
    paragraphs = text.split("\n")
    for item in paragraphs:
        sentences.extend(item.split("."))

    chunks = []
    index = 0
    while True:
        if index > len(sentences):
            break
        if len(sentences[index]) > chunk_size:
            continue

        chunk = ""
        while True:
            if len(chunk) == 0:
                chunk += sentences[index - 1][: -1 * cover_size]
            if len(chunk) >= chunk_size:
                index -= 1
                chunk.replace(sentences[index], "")
                break
            chunk += sentences[index]
            index += 1
        chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    text = """Barton-upon-Humber Post Office is one of 11,500 branches helping customers with access to cash\n\nThe Post Office handled a record amount of cash in July with customers either depositing or withdrawing more than \u00a33.7bn. July\'s record beat previous highs set in May, April and December. The increasing use of the Post Office to handle cash comes as the rate of closure of bank branches shows no sign of slowing. More than 6,000 have shut their doors since 2015, an average of about 50 each month.\n\nBanking hubs are slowly being opened to try to offer the public similar services. There are around 70 already in operation with 100 expected to have opened their doors by Christmas. They aim to provide access to cash for those who need it, and to allow small businesses to deposit takings. But with hubs and post offices unable to provide much more than the basic banking services, the closure of branches is likely to remain contentious for a long time to come.\n\nMairi Wingate says a lot of her customers say cash helps them budget better\n\nMairi Wingate has been the postmistress in Barton-upon-Humber for 19 years and says there are lots of reasons people prefer cash. "Well it can range from anything from people wanting \u00a310 to go to the local hairdressers to go and get their hair done ranging up to \u00a3300 or \u00a3400 and it\'s just budgeting for the weekly shopping and bills. "They know exactly what they\'ve got and they can\'t overspend what they haven\'t got." The Post Office\'s reputation has been seriously damaged in recent years because of the Horizon IT scandal which saw hundreds of postmasters and mistress wrongly convicted of stealing money. During that time however daily visits to Post Office branches have remained relatively stable at around 10 million each week, and its position on the High Street means it offers a local alternative to many people when bank branches shut down.\n\n"Cash is king" according to John, who rarely uses his cards\n\nJohn calls into the Post Office nearly every week to withdraw cash from his bank account. "\u00a350 I\'ve taken out to go shopping with, like I have done for the last 50 years. Cash is still king in my book. "I do have cards but very rarely use them. You can flash a card around without really knowing what you\'re doing," he says. "Cash in the pocket [though] you know where you are."""

    chunks = text_spliter(text, 128, 32)
    print(chunks)
