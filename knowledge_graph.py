from neo4j import GraphDatabase
from datetime import datetime
import os
import json
from init_neo4j import init_neo4j_schema

class KnowledgeGraph:
    def __init__(self, force_init=False):
        """
        Initialize Neo4j connection using environment variables.
        
        Args:
            force_init: If True, force schema reinitialization
        """
        # Set connected to False initially
        self.connected = False
        
        uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            print(f"Attempting to connect to Neo4j at {uri}")
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            
            # Check if schema needs initialization
            relationships_exist = self._check_relationships_exist()
            
            if not relationships_exist or force_init:
                print("Neo4j schema not fully initialized. Running initialization...")
                success = init_neo4j_schema()
                if not success:
                    print("WARNING: Schema initialization failed. Some features may not work.")
            
            # Initialize schema structure (will be skipped if already exists)
            self._init_schema()
            print("Neo4j connection established successfully")
            self.connected = True
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            import traceback
            traceback.print_exc()
            self.connected = False
    
    def _check_relationships_exist(self):
        """Check if the required relationships exist in the database."""
        try:
            with self.driver.session() as session:
                belongs_to = session.run("MATCH ()-[r:BELONGS_TO]->() RETURN count(r) > 0 AS exists").single()["exists"]
                categorized_under = session.run("MATCH ()-[r:CATEGORIZED_UNDER]->() RETURN count(r) > 0 AS exists").single()["exists"]
                
                return belongs_to and categorized_under
        except Exception as e:
            print(f"Error checking relationships: {e}")
            return False
    
    def close(self):
        """Close the Neo4j driver connection."""
        if hasattr(self, 'driver'):
            self.driver.close()
            print("Neo4j connection closed")

    def _init_schema(self):
        """Initialize the knowledge graph schema with main categories."""
        with self.driver.session() as session:
            # Create constraint for unique Issue names
            session.run("""
                CREATE CONSTRAINT issue_name IF NOT EXISTS
                FOR (i:Issue) REQUIRE i.name IS UNIQUE
            """)
            
            # Create constraint for unique SubIssue names
            session.run("""
                CREATE CONSTRAINT subissue_name IF NOT EXISTS
                FOR (s:SubIssue) REQUIRE s.name IS UNIQUE
            """)
            
            # Create constraint for unique Question IDs
            session.run("""
                CREATE CONSTRAINT question_id IF NOT EXISTS
                FOR (q:Question) REQUIRE q.id IS UNIQUE
            """)

            # Create main issue categories
            main_issues = [
                "UI Issue",
                "Payment Issue", 
                "Account Issue",
                "Uncategorized"
            ]
            
            # Create sub-issues with keywords for better classification
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
            
            # Store keywords for classification
            self.classification_data = sub_issues
            
            # Create Issues and SubIssues
            for issue in main_issues:
                session.run("""
                    MERGE (i:Issue {name: $name})
                """, name=issue)
                
                # Create SubIssues and relationships
                for sub_issue in sub_issues[issue]:
                    sub_name = sub_issue["name"]
                    keywords = sub_issue["keywords"]
                    
                    session.run("""
                        MERGE (s:SubIssue {name: $sub_name})
                        SET s.keywords = $keywords
                        MERGE (i:Issue {name: $issue_name})
                        MERGE (s)-[:BELONGS_TO]->(i)
                    """, sub_name=sub_name, keywords=keywords, issue_name=issue)
            
            print(f"Initialized knowledge graph schema with {len(main_issues)} issues and multiple sub-issues")

    def classify_message(self, message):
        """
        Classify a message into appropriate categories using keyword matching.
        Returns tuple of (sub_issue, main_issue)
        """
        if not message or not hasattr(self, 'classification_data'):
            return "General Inquiry", "Uncategorized"
            
        lower = message.lower()
        
        # Score each sub-issue based on keyword matches
        best_score = 0
        best_sub_issue = "General Inquiry"
        best_main_issue = "Uncategorized"
        
        # For each main issue and its sub-issues
        for main_issue, sub_issues in self.classification_data.items():
            for sub_issue in sub_issues:
                sub_name = sub_issue["name"]
                keywords = sub_issue["keywords"]
                
                # Skip empty keyword lists
                if not keywords:
                    continue
                    
                # Count keyword matches
                score = sum(1 for keyword in keywords if keyword in lower)
                
                # If better than current best, update
                if score > best_score:
                    best_score = score
                    best_sub_issue = sub_name
                    best_main_issue = main_issue
        
        # If no matches found, try the simple classification
        if best_score == 0:
            # UI related
            if any(word in lower for word in ["ui", "button", "interface", "layout", "design"]):
                if "button" in lower:
                    return "Button Issue", "UI Issue"
                elif "layout" in lower:
                    return "Layout Issue", "UI Issue"
                return "Interface Problem", "UI Issue"
            
            # Payment related
            elif any(word in lower for word in ["payment", "card", "transaction", "money"]):
                if "card" in lower:
                    return "Card Problem", "Payment Issue"
                elif "transaction" in lower:
                    return "Transaction Error", "Payment Issue"
                return "Payment Processing", "Payment Issue"
            
            # Account related
            elif any(word in lower for word in ["account", "login", "profile", "password", "auth"]):
                if "login" in lower:
                    return "Login Problem", "Account Issue"
                elif "profile" in lower:
                    return "Profile Issue", "Account Issue"
                return "Authentication Error", "Account Issue"
            
            return "General Inquiry", "Uncategorized"
            
        return best_sub_issue, best_main_issue

    def add_question(self, question_id, content, chat_id=None):
        """
        Add a question to the knowledge graph and categorize it.
        
        Args:
            question_id: Unique identifier for the question
            content: The question content
            chat_id: Optional chat session ID
        """
        if not self.connected:
            print("Not connected to Neo4j, cannot add question")
            return False
            
        sub_issue, main_issue = self.classify_message(content)
        
        try:
            with self.driver.session() as session:
                # Create the question node and relationships
                result = session.run("""
                    MERGE (q:Question {id: $qid})
                    SET q.content = $content,
                        q.timestamp = $timestamp,
                        q.chat_id = $chat_id
                    WITH q
                    MATCH (s:SubIssue {name: $sub_issue})-[:BELONGS_TO]->(i:Issue {name: $main_issue})
                    MERGE (q)-[:CATEGORIZED_UNDER]->(s)
                    RETURN q.id as id
                """, qid=question_id, 
                    content=content,
                    timestamp=datetime.now().isoformat(),
                    chat_id=chat_id,
                    sub_issue=sub_issue,
                    main_issue=main_issue)
                
                result_data = result.single()
                print(f"Added question '{question_id}' to Neo4j under category {main_issue} → {sub_issue}")
                return True
        except Exception as e:
            print(f"Error adding question to Neo4j: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_similar_questions(self, category, limit=5):
        """
        Retrieve similar questions from the same category.
        
        Args:
            category: The category (SubIssue) to search in
            limit: Maximum number of questions to return
            
        Returns:
            List of questions with their content
        """
        if not self.connected:
            return []
            
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (q:Question)-[:CATEGORIZED_UNDER]->(s:SubIssue {name: $category})
                    RETURN q.content as content, q.timestamp as timestamp, q.id as id
                    ORDER BY q.timestamp DESC
                    LIMIT $limit
                """, category=category, limit=limit)
                
                return [{"content": record["content"], 
                        "timestamp": record["timestamp"],
                        "id": record["id"]} 
                       for record in result]
        except Exception as e:
            print(f"Error getting similar questions: {e}")
            return []

    def get_category_statistics(self):
        """
        Get statistics about question distribution across categories.
        
        Returns:
            Dictionary with category counts
        """
        if not self.connected:
            return {}
            
        try:
            with self.driver.session() as session:
                # First check if the relationships exist to avoid cryptic errors
                if not self._check_relationships_exist():
                    print("WARNING: Required relationships don't exist. Cannot get statistics.")
                    return {}
                
                result = session.run("""
                    MATCH (q:Question)-[:CATEGORIZED_UNDER]->(s:SubIssue)-[:BELONGS_TO]->(i:Issue)
                    RETURN i.name as issue, s.name as subissue, count(q) as count
                    ORDER BY count DESC
                """)
                
                stats = {}
                for record in result:
                    issue = record["issue"]
                    subissue = record["subissue"]
                    count = record["count"]
                    
                    if issue not in stats:
                        stats[issue] = {"total": 0, "subissues": {}}
                    
                    stats[issue]["total"] += count
                    stats[issue]["subissues"][subissue] = count
                    
                # If no data returned, initialize with empty values
                if not stats:
                    # Get all issue categories
                    categories = session.run("MATCH (i:Issue) RETURN i.name as name")
                    for record in categories:
                        stats[record["name"]] = {"total": 0, "subissues": {}}
                        
                        # Get subcategories for this issue
                        subcategories = session.run("""
                            MATCH (s:SubIssue)-[:BELONGS_TO]->(i:Issue {name: $name})
                            RETURN s.name as name
                        """, name=record["name"])
                        
                        for subcat in subcategories:
                            stats[record["name"]]["subissues"][subcat["name"]] = 0
                
                return stats
        except Exception as e:
            print(f"Error getting category statistics: {e}")
            import traceback
            traceback.print_exc()
            return {}
            
    def get_knowledge_graph_data(self):
        """
        Export knowledge graph data for visualization.
        
        Returns:
            Dictionary with nodes and links for visualization
        """
        if not self.connected:
            return {"nodes": [], "links": []}
            
        try:
            with self.driver.session() as session:
                # Get all nodes
                nodes_result = session.run("""
                    MATCH (n)
                    WHERE n:Issue OR n:SubIssue OR n:Question
                    RETURN 
                        id(n) as id,
                        labels(n) as labels,
                        CASE
                            WHEN 'Issue' IN labels(n) THEN n.name
                            WHEN 'SubIssue' IN labels(n) THEN n.name
                            WHEN 'Question' IN labels(n) THEN LEFT(n.content, 50) + '...'
                            ELSE 'Unknown'
                        END as name,
                        CASE
                            WHEN 'Issue' IN labels(n) THEN 'issue'
                            WHEN 'SubIssue' IN labels(n) THEN 'subissue'
                            WHEN 'Question' IN labels(n) THEN 'question'
                            ELSE 'unknown'
                        END as type,
                        CASE
                            WHEN n:Question THEN n.content
                            ELSE null
                        END as content,
                        CASE
                            WHEN n:Question THEN n.timestamp
                            ELSE null
                        END as timestamp
                """)
                
                # Get all relationships
                links_result = session.run("""
                    MATCH (a)-[r]->(b)
                    WHERE (a:Issue OR a:SubIssue OR a:Question) AND (b:Issue OR b:SubIssue OR b:Question)
                    RETURN 
                        id(a) as source,
                        id(b) as target,
                        type(r) as type
                """)
                
                # Convert to visualization format
                nodes = [
                    {
                        "id": record["id"],
                        "name": record["name"],
                        "labels": record["labels"],
                        "type": record["type"],
                        "content": record["content"],
                        "timestamp": record["timestamp"]
                    }
                    for record in nodes_result
                ]
                
                links = [
                    {
                        "source": record["source"],
                        "target": record["target"],
                        "type": record["type"]
                    }
                    for record in links_result
                ]
                
                return {"nodes": nodes, "links": links}
        except Exception as e:
            print(f"Error getting knowledge graph data: {e}")
            return {"nodes": [], "links": []}
            
    def reclassify_question(self, question_id, new_sub_issue):
        """
        Reclassify a question to a different sub-issue.
        
        Args:
            question_id: ID of the question to reclassify
            new_sub_issue: Name of the new sub-issue
            
        Returns:
            Boolean indicating success
        """
        if not self.connected:
            return False
            
        try:
            with self.driver.session() as session:
                # Get the main issue for the sub-issue
                main_issue_result = session.run("""
                    MATCH (s:SubIssue {name: $sub_issue})-[:BELONGS_TO]->(i:Issue)
                    RETURN i.name as main_issue
                """, sub_issue=new_sub_issue)
                
                record = main_issue_result.single()
                if not record:
                    print(f"Error: Could not find main issue for sub-issue {new_sub_issue}")
                    return False
                    
                main_issue = record["main_issue"]
                
                # Update the relationship
                session.run("""
                    MATCH (q:Question {id: $qid})-[r:CATEGORIZED_UNDER]->(:SubIssue)
                    DELETE r
                    WITH q
                    MATCH (s:SubIssue {name: $sub_issue})
                    MERGE (q)-[:CATEGORIZED_UNDER]->(s)
                """, qid=question_id, sub_issue=new_sub_issue)
                
                print(f"Reclassified question {question_id} to {main_issue} → {new_sub_issue}")
                return True
        except Exception as e:
            print(f"Error reclassifying question: {e}")
            return False
            
    def get_graph_visualization_data(self):
        """
        Get data specifically formatted for visualization libraries.
        Returns a JSON string that can be used with visualization libraries.
        """
        if not self.connected:
            return json.dumps({"nodes": [], "links": []})
            
        graph_data = self.get_knowledge_graph_data()
        
        # Format for visualization libraries like d3.js
        vis_nodes = []
        for node in graph_data["nodes"]:
            # Set node colors based on type
            color = "#1f77b4"  # default blue
            if "issue" in node["type"]:
                color = "#ff7f0e"  # orange for issues
            elif "subissue" in node["type"]:
                color = "#2ca02c"  # green for subissues
                
            vis_nodes.append({
                "id": node["id"],
                "label": node["name"],
                "group": node["type"],
                "color": color,
                "size": 15 if node["type"] == "issue" else (10 if node["type"] == "subissue" else 5)
            })
        
        vis_links = []
        for link in graph_data["links"]:
            vis_links.append({
                "source": link["source"],
                "target": link["target"],
                "value": 1,
                "label": link["type"]
            })
            
        return json.dumps({"nodes": vis_nodes, "links": vis_links})

    def check_connection(self):
        """Check if the connection to Neo4j is still active."""
        if not hasattr(self, 'driver'):
            self.connected = False
            return False
            
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            self.connected = True
            return True
        except Exception as e:
            print(f"Neo4j connection check failed: {e}")
            self.connected = False
            return False 