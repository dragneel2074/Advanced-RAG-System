from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_neo4j_schema():
    """
    Initialize the Neo4j database schema with required nodes and relationships.
    This ensures the relationships exist before querying them.
    """
    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    print(f"Connecting to Neo4j at {uri}...")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            result.single()
            print("Connection successful!")
            
            # Create constraints
            print("Creating constraints...")
            session.run("""
                CREATE CONSTRAINT issue_name IF NOT EXISTS
                FOR (i:Issue) REQUIRE i.name IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT subissue_name IF NOT EXISTS
                FOR (s:SubIssue) REQUIRE s.name IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT question_id IF NOT EXISTS
                FOR (q:Question) REQUIRE q.id IS UNIQUE
            """)
            
            # Define issue categories
            main_issues = [
                "UI Issue",
                "Payment Issue", 
                "Account Issue",
                "Uncategorized"
            ]
            
            # Define sub-issues with keywords
            sub_issues = {
                "UI Issue": [
                    {"name": "Interface Problem", "keywords": ["ui", "interface", "display", "screen", "view"]},
                    {"name": "Button Issue", "keywords": ["button", "click", "press", "tap", "select"]},
                    {"name": "Layout Issue", "keywords": ["layout", "position", "alignment", "design", "responsive"]}
                ],
                "Payment Issue": [
                    {"name": "Transaction Error", "keywords": ["transaction", "error", "failed", "payment error"]},
                    {"name": "Card Problem", "keywords": ["card", "credit", "debit", "expired", "declined"]},
                    {"name": "Payment Processing", "keywords": ["process", "payment", "checkout", "pay", "purchase"]}
                ],
                "Account Issue": [
                    {"name": "Login Problem", "keywords": ["login", "sign in", "cannot access", "password", "username"]},
                    {"name": "Profile Issue", "keywords": ["profile", "account", "settings", "preferences", "personal"]},
                    {"name": "Authentication Error", "keywords": ["auth", "token", "verify", "2fa", "authentication"]}
                ],
                "Uncategorized": [
                    {"name": "General Inquiry", "keywords": []},
                    {"name": "Other", "keywords": []}
                ]
            }
            
            # Create Issues
            print("Creating issue nodes...")
            for issue in main_issues:
                session.run("""
                    MERGE (i:Issue {name: $name})
                """, name=issue)
                
                # Create SubIssues and BELONGS_TO relationships
                for sub_issue in sub_issues[issue]:
                    sub_name = sub_issue["name"]
                    keywords = sub_issue["keywords"]
                    
                    print(f"Creating {sub_name} as part of {issue}")
                    
                    session.run("""
                        MERGE (s:SubIssue {name: $sub_name})
                        SET s.keywords = $keywords
                        WITH s
                        MATCH (i:Issue {name: $issue_name})
                        MERGE (s)-[:BELONGS_TO]->(i)
                    """, sub_name=sub_name, keywords=keywords, issue_name=issue)
            
            # Create a sample question to establish the CATEGORIZED_UNDER relationship
            print("Creating sample question for relationship establishment...")
            session.run("""
                MERGE (q:Question {id: 'sample-question-id'})
                SET q.content = 'This is a sample question to establish the relationship structure',
                    q.timestamp = datetime()
                WITH q
                MATCH (s:SubIssue {name: 'General Inquiry'})
                MERGE (q)-[:CATEGORIZED_UNDER]->(s)
            """)
            
            # Verify relationships exist
            print("Verifying relationships...")
            belongs_to_result = session.run("MATCH ()-[r:BELONGS_TO]->() RETURN count(r) AS count").single()
            categorized_under_result = session.run("MATCH ()-[r:CATEGORIZED_UNDER]->() RETURN count(r) AS count").single()
            
            print(f"BELONGS_TO relationships: {belongs_to_result['count']}")
            print(f"CATEGORIZED_UNDER relationships: {categorized_under_result['count']}")
            
            print("Schema initialization complete!")
        
        driver.close()
        return True
    except Exception as e:
        print(f"Error initializing Neo4j schema: {e}")
        import traceback
        traceback.print_exc()
        if 'driver' in locals():
            driver.close()
        return False

if __name__ == "__main__":
    # Run the initialization when script is executed directly
    init_neo4j_schema() 