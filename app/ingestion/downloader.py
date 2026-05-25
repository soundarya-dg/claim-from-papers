import os
import time
from pathlib import Path
from typing import List, Optional
import arxiv
import requests
from datetime import datetime


class ArxivDownloader:
    """Downloads research papers from arXiv based on search query and date range"""
    
    def __init__(
        self,
        output_dir: str = "data/papers",
        max_results: int = 30, # Maximum number of papers to download
        start_year: int = 2024,
        end_year: int = 2026
    ):
        self.output_dir = Path(output_dir)
        self.max_results = max_results
        self.start_year = start_year
        self.end_year = end_year
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def is_within_date_range(self, published_date: datetime) -> bool:
        """Check if paper was published within the specified date range"""
        year = published_date.year
        return self.start_year <= year <= self.end_year
    
    def clean_filename(self, filename: str) -> str:
        """Sanitize filename to remove invalid characters"""
        # Replace invalid characters with underscore
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        # Limit filename length
        return filename[:200]
    
    def download_papers(self, query: str, category: Optional[str] = None) -> List[str]:
        """
        Download papers from arXiv based on search query.
        
        Args:
            query: Search query string
            category: ArXiv category to search in (e.g., 'cs.CL', 'cs.AI')
            
        Returns:
            List of downloaded file paths
        """
        print(f"\n{'-'*80}")
        print(f"ArXiv Paper Downloader")
        print(f"{'-'*80}")
        print(f"Query: {query}")
        print(f"Date Range: {self.start_year} - {self.end_year}")
        print(f"Target: {self.max_results} papers")
        print(f"Output: {self.output_dir}")
        print(f"{'-'*80}\n")
        
        # Construct search query with category if provided
        search_query = f"{query}"
        if category:
            search_query = f"cat:{category} AND {query}"
        
        # Create arxiv search client
        client = arxiv.Client()
        
        # Search for papers
        search = arxiv.Search(
            query=search_query,
            max_results=self.max_results * 3,  # Get more to filter by date
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        downloaded_files = []
        downloaded_count = 0
        skipped_count = 0
        
        print("Searching and downloading papers...\n")
        
        try:
            for result in client.results(search):
                # Check if we have downloaded enough papers
                if downloaded_count >= self.max_results:
                    break
                
                # Check if paper is within date range
                if not self.is_within_date_range(result.published):
                    skipped_count += 1
                    continue
                
                # Create filename from paper ID and title
                paper_id = result.entry_id.split('/')[-1]
                title_part = self.clean_filename(result.title[:50])
                filename = f"{paper_id}_{title_part}.pdf"
                filepath = self.output_dir / filename
                
                # Skip if already downloaded
                if filepath.exists():
                    print(f"[SKIP] Already exists: {filename}")
                    downloaded_count += 1
                    downloaded_files.append(str(filepath))
                    continue
                
                try:
                    # Download the paper
                    print(f"[{downloaded_count + 1}/{self.max_results}] Downloading: {result.title}")
                    print(f"    Author(s): {', '.join([a.name for a in result.authors[:3]])}{'...' if len(result.authors) > 3 else ''}")
                    print(f"    Published: {result.published.strftime('%Y-%m-%d')}")
                    print(f"    URL: {result.entry_id}")
                    
                    # Download PDF using the correct method
                    pdf_url = result.pdf_url
                    response = requests.get(pdf_url)
                    response.raise_for_status()
                    
                    # Save PDF
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    downloaded_files.append(str(filepath))
                    downloaded_count += 1
                    
                    print(f"    Saved to: {filepath.name}\n")
                    
                    # Add delay
                    time.sleep(3)
                    
                except Exception as e:
                    print(f"    Error downloading: {e}\n")
                    continue
                    
        except KeyboardInterrupt:
            print("\n\nDownload interrupted by user.")
        except Exception as e:
            print(f"\nError during search/download: {e}")
        
        # Print summary
        print(f"{'-'*80}")
        print(f"Successfully downloaded: {downloaded_count} papers")
        print(f"Skipped (out of date range): {skipped_count} papers")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"{'-'*80}\n")
        
        return downloaded_files


def main():
    current_year = datetime.now().year
    downloader = ArxivDownloader(
        output_dir="data/papers",
        max_results=30,
        start_year=2024,
        end_year=current_year  # Use current year instead of fixed 2025
    )
    
    # Download papers on the topic
    query = "AI-Generated Text Detection OR LLM-Generated Text Detection"
    
    downloaded_files = downloader.download_papers(
        query=query,
        category="cs.CL"  # Computation and Language category
    )
    
    if downloaded_files:
        print(f"  Successfully downloaded {len(downloaded_files)} papers!")
    else:
        print(" No papers were downloaded.")


if __name__ == "__main__":
    main()