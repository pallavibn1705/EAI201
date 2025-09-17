import asyncio
from browser_use import Agent  # Correct import
from browser_use.llm import ChatOpenAI  # For the LLM model
from browser_use.browser import BrowserSession  # If you want to define a session
from dotenv import load_dotenv
import os

load_dotenv()
# ensure you have OPENAI_API_KEY in your .env

async def main():
    # Optionally set up a browser session if needed
    # session = BrowserSession(...)  

    # Define your task
    task = """
    You need to perform the following browser navigation related to the academic topic 'Differential Equations':

    1. Open https://en.wikipedia.org in the browser.
    2. Search for "Differential Equations".
    3. Click on the main article for "Differential Equation".
    4. From there, follow the hyperlink to "Partial Differential Equation".
    5. Then, navigate to the page on "Heat Equation" (a common PDE).
    6. Next, go to the linked page on "Fourier Series" used in solving PDEs.
    7. After that, visit the article about "Laplace Transform" (important in ODEs and PDEs).
    8. Finally, navigate to the "Runge–Kutta methods" page for numerical solutions.
    """

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o")  # Or another supported model

    agent = Agent(
        task=task,
        llm=llm,
        # If you created a session, pass it:
        # browser_session=session
    )

    result = await agent.run()
    print("Task completed. Result:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
