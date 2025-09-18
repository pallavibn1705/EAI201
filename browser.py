import asyncio
from browser_use import Agent 
from browser_use.llm import ChatOpenAI  
from dotenv import load_dotenv


load_dotenv()


async def main():
   
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

   
    llm = ChatOpenAI(model="gpt-4o")  

    agent = Agent(
        task=task,
        llm=llm,
      
    )

    result = await agent.run()
    print("Task completed. Result:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

