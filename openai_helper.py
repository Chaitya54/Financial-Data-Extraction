from openai import OpenAI
from secret_key import openai_key
import json
import pandas as pd

client = OpenAI(api_key=openai_key)


def extract_financial_data(text):
    prompt = get_prompt_financial() + text

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    content = response.choices[0].message.content

    try:
        data = json.loads(content)
        return pd.DataFrame(data.items(), columns=["Measure", "Value"])

    except (json.JSONDecodeError, IndexError):
        pass
    return pd.DataFrame({
        "Measure": ["Company Name", "Stock Symbol", "Revenue", "Net Income", "EPS"],
        "Value": ["", "", "", "", ""]
    })


def get_prompt_financial():
    return '''Please retrieve company name, revenue, net income and earnings per share(a.k.a. EPS) 
    from the following news article. If you can't find the information from the article then return "".
    Do not make things up.
    Then retrieve a stock symbol corresponding to that company. For this you can use your general knowledge
    (it doesn't have to be from this article). Always return your response as a valid JSON string. The 
    format should be this,
    {
        "Company Name": "Walmart",
        "Stock Symbol": "WMT",
        "Revenue": ""12.34 million",
        "Net Income": "34.78 million",
        "EPS": "2.1 $"
    }
    News Article
    ============

    '''


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    text = '''
    Apple Inc. (AAPL) has announced its fiscal results for the first quarter of 2024, demonstrating robust financial performance across key metrics, including revenue, net income, and earnings per share (EPS).
    The tech giant reported a staggering $150 billion in revenue for the quarter, representing a notable 25% increase compared to the same period last year. This remarkable figure underscores Apple's continued ability to drive substantial sales despite ongoing economic challenges and geopolitical uncertainties.
    Furthermore, Apple disclosed a net income of $30 billion for the quarter, reflecting the company's solid profitability and operational efficiency. This figure exceeded analyst expectations and reinforces Apple's position as one of the most profitable companies in the world.
    In terms of earnings per share (EPS), Apple reported an impressive $7.50, surpassing both consensus estimates and the company's own projections. This strong EPS performance indicates the company's ability to deliver value to its shareholders through sustained growth and financial prudence.
    Tim Cook, CEO of Apple, expressed his satisfaction with the company's fiscal performance, stating, "We are thrilled with our record-breaking results for the first quarter. This achievement is a testament to the dedication and innovation of our teams worldwide, as well as the enduring appeal of our products and services."
    Apple's stellar financial results were driven by robust sales of its flagship product, the iPhone, as well as strong performance in its services and wearables segments. The company's ecosystem of devices and services continues to resonate with consumers globally, contributing to its sustained growth trajectory.
    Looking ahead, Apple remains optimistic about its future prospects, with a promising lineup of new products and services in the pipeline. The company is set to unveil innovative offerings later this year, including the highly anticipated iPhone 15 and advancements in augmented reality technology.
    Investors responded positively to Apple's fiscal report, with the company's stock price experiencing a surge in after-hours trading. Analysts remain bullish on Apple's long-term outlook, citing its strong financial fundamentals and strategic initiatives aimed at driving continued growth and innovation.
    '''

    df = extract_financial_data(text)
    print(df.to_string())
