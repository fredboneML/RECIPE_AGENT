import re
import logging

logger = logging.getLogger(__name__)


class AgentManager:
    def _sanitize_sql_query(self, sql_query: str) -> str:
        """Sanitize SQL query to fix common issues"""
        try:
            # Remove any markdown formatting
            if sql_query.startswith("```") and sql_query.endswith("```"):
                sql_query = sql_query[3:-3].strip()
            elif sql_query.startswith("```sql") and sql_query.endswith("```"):
                sql_query = sql_query[6:-3].strip()
            elif sql_query.startswith("```"):
                # Handle case where only opening ``` is present
                sql_query = sql_query[3:].strip()

            # Remove any remaining markdown formatting
            sql_query = re.sub(r'^```sql\s*', '', sql_query)
            sql_query = re.sub(r'^```\s*', '', sql_query)
            sql_query = re.sub(r'\s*```$', '', sql_query)

            if sql_query.startswith("sql"):
                sql_query = sql_query[3:].strip()

            # Fix issue with semicolons before LIMIT
            sql_query = sql_query.replace("; LIMIT", " LIMIT")

            # Remove any trailing semicolons
            sql_query = sql_query.rstrip(";")

            # Ensure there's only one LIMIT clause at the end
            if "LIMIT" in sql_query:
                # Split by LIMIT, keeping only the first part and the last LIMIT clause
                parts = sql_query.split("LIMIT")
                if len(parts) > 2:
                    sql_query = parts[0] + "LIMIT" + parts[-1]
            else:
                # Add LIMIT 20 if no LIMIT clause is present
                sql_query = f"{sql_query} LIMIT 20"

            # Convert exact matches for phone numbers to LIKE statements
            # Look for patterns like: telephone_number = '1234567890' or telephone_number='1234567890'
            phone_pattern = re.compile(
                r"(telephone_number\s*=\s*['\"]([\d\+]+)['\"])")
            matches = phone_pattern.findall(sql_query)
            for match, phone_number in matches:
                replacement = f"telephone_number LIKE '%{phone_number}%'"
                sql_query = sql_query.replace(match, replacement)

            # Convert exact matches for other common entity fields to LIKE statements
            entity_fields = ['name', 'address', 'email',
                             'clid', 'customer_id', 'account_number']
            for field in entity_fields:
                # Match pattern: field = 'value' or field='value'
                field_pattern = re.compile(
                    f"({field}\\s*=\\s*['\"](.*?)['\"])")
                matches = field_pattern.findall(sql_query)
                for match, value in matches:
                    replacement = f"{field} LIKE '%{value}%'"
                    sql_query = sql_query.replace(match, replacement)

            # Remove any explanatory text after the SQL query
            # Look for common patterns like "This query will..." or "The query..."
            explanatory_pattern = re.compile(
                r'```\s*(This query|The query|This SQL|The SQL).*$', re.DOTALL | re.IGNORECASE)
            sql_query = explanatory_pattern.sub('', sql_query).strip()

            # Remove any remaining backticks
            sql_query = sql_query.replace('```', '').strip()

            return sql_query
        except Exception as e:
            logger.error(f"Error sanitizing SQL query: {e}")
            logger.exception("Detailed error:")
            return sql_query  # Return original query if sanitization fails

    def generate_sql_query(self, sql_query):
        try:
            # Clean up the SQL query
            sql_query = self._sanitize_sql_query(sql_query)

            # Ensure LIMIT 20 is present
            if not re.search(r'\bLIMIT\s+\d+', sql_query, re.IGNORECASE):
                # Remove any trailing semicolon
                sql_query = sql_query.rstrip(';').strip()
                # Add LIMIT 20 to the end
                sql_query = f"{sql_query} LIMIT 20"

            return sql_query
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            logger.exception("Detailed error:")
            raise ValueError(f"Error generating SQL query: {str(e)}")
