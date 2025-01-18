import asyncio
from crawl4ai import AsyncWebCrawler, CacheMode

async def main():
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url="https://www.nbcnews.com/business")
        # Soone will be change to result.markdown
        print(result.markdown_v2.raw_markdown) 

if __name__ == "__main__":
    asyncio.run(main())
