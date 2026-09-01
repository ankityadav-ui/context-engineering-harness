from neo4j import GraphDatabase

from .config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
)


class GraphRAG:

    def __init__(self):
        """
        Initialize Neo4j Graph RAG connection.
        """

        uri = NEO4J_URI

        # ----------------------------------------------------
        # Neo4j Aura SSL certificate workaround
        # ----------------------------------------------------

        # neo4j+s:// performs certificate verification.
        #
        # On this Windows setup, certificate verification
        # causes SSLCertVerificationError.
        #
        # neo4j+ssc:// keeps the connection encrypted while
        # allowing the self-signed certificate.

        if uri.startswith("neo4j+s://"):
            uri = uri.replace(
                "neo4j+s://",
                "neo4j+ssc://",
                1,
            )

        self.driver = GraphDatabase.driver(
            uri,
            auth=(
                NEO4J_USERNAME,
                NEO4J_PASSWORD,
            ),
        )

        self.database = NEO4J_DATABASE

    # ========================================================
    # CONNECTION
    # ========================================================

    def verify_connection(self):
        """
        Verify Neo4j connectivity and authentication.
        """

        self.driver.verify_connectivity()

        return True

    # ========================================================
    # GENERIC CYPHER QUERY
    # ========================================================

    def run_query(
        self,
        query,
        parameters=None,
    ):
        """
        Execute a Cypher query and return results as dictionaries.
        """

        with self.driver.session(
            database=self.database
        ) as session:

            result = session.run(
                query,
                parameters or {},
            )

            return result.data()

    # ========================================================
    # CREATE COURSE GRAPH
    # ========================================================

    def create_course_graph(self):
        """
        Create the initial Machine Learning knowledge graph.

        Graph structure:

            Course
              |
              |-- HAS_CODE ------> CourseCode
              |
              |-- HAS_SEMESTER --> Semester
              |
              |-- HAS_BRANCH ----> CSEBranch
              |
              |-- HAS_TOPIC -----> Topic

        MERGE prevents duplicate nodes and relationships.
        """

        query = """
        MERGE (course:Course {
            name: "Machine Learning"
        })

        MERGE (code:CourseCode {
            value: "CSL422"
        })

        MERGE (semester:Semester {
            value: "VI"
        })

        MERGE (branch:CSEBranch {
            value: "CSE"
        })

        MERGE (topic:Topic {
            name: "Machine Learning"
        })

        MERGE (course)-[:HAS_CODE]->(code)

        MERGE (course)-[:HAS_SEMESTER]->(semester)

        MERGE (course)-[:HAS_BRANCH]->(branch)

        MERGE (course)-[:HAS_TOPIC]->(topic)

        RETURN
            course.name AS course,
            code.value AS code,
            semester.value AS semester,
            branch.value AS branch,
            topic.name AS topic
        """

        return self.run_query(query)

    # ========================================================
    # FIND COURSE NAME
    # ========================================================

    def find_course(self, query_text):
        """
        Find a course whose name or topic appears inside
        the natural-language query.

        Example:

            "What is the course code of Machine Learning?"

        will identify:

            Machine Learning
        """

        cypher = """
        MATCH (course:Course)

        OPTIONAL MATCH
            (course)-[:HAS_TOPIC]->(topic:Topic)

        WITH
            course,
            topic,
            toLower($query) AS q

        WHERE
            q CONTAINS toLower(course.name)
            OR
            q CONTAINS toLower(coalesce(topic.name, ""))

        RETURN
            course.name AS course
        """

        return self.run_query(
            cypher,
            {
                "query": query_text,
            },
        )

    # ========================================================
    # SEARCH COURSE
    # ========================================================

    def search_course(
        self,
        query_text,
    ):
        """
        Search the knowledge graph using natural-language queries.

        The method first identifies the course mentioned in the
        query and then retrieves the properties requested by
        the user.

        Examples:

            "Machine Learning"

            "What is the course code of Machine Learning?"

            "What semester is Machine Learning?"

            "What is the course code and semester of Machine Learning?"

            "Which branch offers Machine Learning?"
        """

        query_text = query_text.strip()

        if not query_text:
            return []

        q = query_text.lower()

        # ====================================================
        # IDENTIFY COURSE
        # ====================================================

        courses = self.find_course(
            query_text
        )

        # ----------------------------------------------------
        # If no course is explicitly identified,
        # fall back to general graph search.
        # ----------------------------------------------------

        if not courses:

            cypher = """
            MATCH (course:Course)

            OPTIONAL MATCH
                (course)-[:HAS_CODE]->(code:CourseCode)

            OPTIONAL MATCH
                (course)-[:HAS_SEMESTER]->(semester:Semester)

            OPTIONAL MATCH
                (course)-[:HAS_BRANCH]->(branch:CSEBranch)

            OPTIONAL MATCH
                (course)-[:HAS_TOPIC]->(topic:Topic)

            WITH
                course,
                code,
                semester,
                branch,
                topic

            WHERE
                toLower(course.name)
                    CONTAINS toLower($query)

                OR

                toLower(
                    coalesce(topic.name, "")
                )
                    CONTAINS toLower($query)

            RETURN
                course.name AS course,
                code.value AS course_code,
                semester.value AS semester,
                branch.value AS branch,
                topic.name AS topic
            """

            return self.run_query(
                cypher,
                {
                    "query": query_text,
                },
            )

        # ====================================================
        # EXTRACT COURSE NAMES
        # ====================================================

        course_names = [
            item["course"]
            for item in courses
            if item.get("course")
        ]

        if not course_names:
            return []

        # ====================================================
        # COURSE CODE + SEMESTER
        # ====================================================

        if (
            "course code" in q
            and "semester" in q
        ):

            cypher = """
            MATCH (course:Course)
            WHERE course.name IN $courses

            OPTIONAL MATCH
                (course)-[:HAS_CODE]->(code:CourseCode)

            OPTIONAL MATCH
                (course)-[:HAS_SEMESTER]->(semester:Semester)

            RETURN
                course.name AS course,
                code.value AS course_code,
                semester.value AS semester
            """

            return self.run_query(
                cypher,
                {
                    "courses": course_names,
                },
            )

        # ====================================================
        # COURSE CODE
        # ====================================================

        if (
            "course code" in q
            or "course number" in q
            or "code of course" in q
            or "code for course" in q
            or "what is the code" in q
        ):

            cypher = """
            MATCH (course:Course)
            WHERE course.name IN $courses

            MATCH
                (course)-[:HAS_CODE]->
                (code:CourseCode)

            RETURN
                course.name AS course,
                code.value AS course_code
            """

            return self.run_query(
                cypher,
                {
                    "courses": course_names,
                },
            )

        # ====================================================
        # SEMESTER
        # ====================================================

        if (
            "semester" in q
            or "which semester" in q
            or "what semester" in q
        ):

            cypher = """
            MATCH (course:Course)
            WHERE course.name IN $courses

            MATCH
                (course)-[:HAS_SEMESTER]->
                (semester:Semester)

            RETURN
                course.name AS course,
                semester.value AS semester
            """

            return self.run_query(
                cypher,
                {
                    "courses": course_names,
                },
            )

        # ====================================================
        # BRANCH
        # ====================================================

        if (
            "branch" in q
            or "which branch" in q
            or "what branch" in q
        ):

            cypher = """
            MATCH (course:Course)
            WHERE course.name IN $courses

            MATCH
                (course)-[:HAS_BRANCH]->
                (branch:CSEBranch)

            RETURN
                course.name AS course,
                branch.value AS branch
            """

            return self.run_query(
                cypher,
                {
                    "courses": course_names,
                },
            )

        # ====================================================
        # TOPIC
        # ====================================================

        if (
            "topic" in q
            or "topics" in q
        ):

            cypher = """
            MATCH (course:Course)
            WHERE course.name IN $courses

            MATCH
                (course)-[:HAS_TOPIC]->
                (topic:Topic)

            RETURN
                course.name AS course,
                topic.name AS topic
            """

            return self.run_query(
                cypher,
                {
                    "courses": course_names,
                },
            )

        # ====================================================
        # GENERAL COURSE SEARCH
        # ====================================================

        cypher = """
        MATCH (course:Course)
        WHERE course.name IN $courses

        OPTIONAL MATCH
            (course)-[:HAS_CODE]->(code:CourseCode)

        OPTIONAL MATCH
            (course)-[:HAS_SEMESTER]->(semester:Semester)

        OPTIONAL MATCH
            (course)-[:HAS_BRANCH]->(branch:CSEBranch)

        OPTIONAL MATCH
            (course)-[:HAS_TOPIC]->(topic:Topic)

        RETURN
            course.name AS course,
            code.value AS course_code,
            semester.value AS semester,
            branch.value AS branch,
            topic.name AS topic
        """

        return self.run_query(
            cypher,
            {
                "courses": course_names,
            },
        )

    # ========================================================
    # BUILD GRAPH CONTEXT
    # ========================================================

    def build_graph_context(
        self,
        query_text,
    ):
        """
        Retrieve graph information and convert it into
        LLM-friendly context.

        Returns:

            {
                "context": "...",
                "results": [...]
            }
        """

        results = self.search_course(
            query_text
        )

        # ----------------------------------------------------
        # Nothing found
        # ----------------------------------------------------

        if not results:

            return {
                "context": "",
                "results": [],
            }

        # ----------------------------------------------------
        # Convert results into context
        # ----------------------------------------------------

        context_parts = []

        for index, result in enumerate(
            results,
            start=1,
        ):

            context_parts.append(
                f"--- GRAPH RESULT {index} ---"
            )

            # -----------------------------------------------
            # Course
            # -----------------------------------------------

            if result.get("course"):

                context_parts.append(
                    f"Course: {result['course']}"
                )

            # -----------------------------------------------
            # Course code
            # -----------------------------------------------

            if result.get("course_code"):

                context_parts.append(
                    f"Course Code: {result['course_code']}"
                )

            # -----------------------------------------------
            # Semester
            # -----------------------------------------------

            if result.get("semester"):

                context_parts.append(
                    f"Semester: {result['semester']}"
                )

            # -----------------------------------------------
            # Branch
            # -----------------------------------------------

            if result.get("branch"):

                context_parts.append(
                    f"Branch: {result['branch']}"
                )

            # -----------------------------------------------
            # Topic
            # -----------------------------------------------

            if result.get("topic"):

                context_parts.append(
                    f"Topic: {result['topic']}"
                )

            context_parts.append("")

        return {
            "context": "\n".join(
                context_parts
            ).strip(),

            "results": results,
        }

    # ========================================================
    # CLOSE CONNECTION
    # ========================================================

    def close(self):
        """
        Close the Neo4j driver.
        """

        self.driver.close()


# ============================================================
# SINGLE GRAPH RAG INSTANCE
# ============================================================

graph_rag = GraphRAG()