/**
 * GraphQL to Cypher Query Translator Implementation
 */

#include "graphql/query_translator.h"
#include <sstream>
#include <algorithm>
#include <regex>

namespace kuzu {
namespace graphql_agi {

// ============================================================================
// GraphQL Lexer Implementation
// ============================================================================

GraphQLLexer::GraphQLLexer(const std::string& source) : source_(source) {}

void GraphQLLexer::skipWhitespace() {
    while (pos_ < source_.size()) {
        char c = source_[pos_];
        if (c == ' ' || c == '\t' || c == '\r') {
            ++pos_;
            ++column_;
        } else if (c == '\n') {
            ++pos_;
            ++line_;
            column_ = 1;
        } else if (c == '#') {
            skipComment();
        } else if (c == ',' || c == '\xEF' || c == '\xBB' || c == '\xBF') {
            // Skip BOM and insignificant commas
            ++pos_;
            ++column_;
        } else {
            break;
        }
    }
}

void GraphQLLexer::skipComment() {
    while (pos_ < source_.size() && source_[pos_] != '\n') {
        ++pos_;
    }
}

GraphQLLexer::Token GraphQLLexer::readName() {
    Token token;
    token.type = TokenType::Name;
    token.line = line_;
    token.column = column_;

    size_t start = pos_;
    while (pos_ < source_.size()) {
        char c = source_[pos_];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_') {
            ++pos_;
            ++column_;
        } else {
            break;
        }
    }

    token.value = source_.substr(start, pos_ - start);
    return token;
}

GraphQLLexer::Token GraphQLLexer::readNumber() {
    Token token;
    token.line = line_;
    token.column = column_;

    size_t start = pos_;
    bool isFloat = false;

    // Optional negative sign
    if (pos_ < source_.size() && source_[pos_] == '-') {
        ++pos_;
        ++column_;
    }

    // Integer part
    while (pos_ < source_.size() && source_[pos_] >= '0' && source_[pos_] <= '9') {
        ++pos_;
        ++column_;
    }

    // Optional fractional part
    if (pos_ < source_.size() && source_[pos_] == '.') {
        isFloat = true;
        ++pos_;
        ++column_;
        while (pos_ < source_.size() && source_[pos_] >= '0' && source_[pos_] <= '9') {
            ++pos_;
            ++column_;
        }
    }

    // Optional exponent
    if (pos_ < source_.size() && (source_[pos_] == 'e' || source_[pos_] == 'E')) {
        isFloat = true;
        ++pos_;
        ++column_;
        if (pos_ < source_.size() && (source_[pos_] == '+' || source_[pos_] == '-')) {
            ++pos_;
            ++column_;
        }
        while (pos_ < source_.size() && source_[pos_] >= '0' && source_[pos_] <= '9') {
            ++pos_;
            ++column_;
        }
    }

    token.type = isFloat ? TokenType::FloatValue : TokenType::IntValue;
    token.value = source_.substr(start, pos_ - start);
    return token;
}

GraphQLLexer::Token GraphQLLexer::readString() {
    Token token;
    token.type = TokenType::StringValue;
    token.line = line_;
    token.column = column_;

    ++pos_;  // Skip opening quote
    ++column_;

    std::ostringstream value;
    while (pos_ < source_.size()) {
        char c = source_[pos_];
        if (c == '"') {
            ++pos_;
            ++column_;
            break;
        } else if (c == '\\') {
            ++pos_;
            ++column_;
            if (pos_ < source_.size()) {
                char escaped = source_[pos_];
                switch (escaped) {
                    case '"': value << '"'; break;
                    case '\\': value << '\\'; break;
                    case '/': value << '/'; break;
                    case 'b': value << '\b'; break;
                    case 'f': value << '\f'; break;
                    case 'n': value << '\n'; break;
                    case 'r': value << '\r'; break;
                    case 't': value << '\t'; break;
                    default: value << escaped;
                }
                ++pos_;
                ++column_;
            }
        } else if (c == '\n') {
            // Invalid: newline in string
            break;
        } else {
            value << c;
            ++pos_;
            ++column_;
        }
    }

    token.value = value.str();
    return token;
}

GraphQLLexer::Token GraphQLLexer::readBlockString() {
    Token token;
    token.type = TokenType::BlockStringValue;
    token.line = line_;
    token.column = column_;

    pos_ += 3;  // Skip opening """
    column_ += 3;

    std::ostringstream value;
    while (pos_ + 2 < source_.size()) {
        if (source_[pos_] == '"' && source_[pos_ + 1] == '"' && source_[pos_ + 2] == '"') {
            pos_ += 3;
            column_ += 3;
            break;
        }

        if (source_[pos_] == '\n') {
            value << '\n';
            ++pos_;
            ++line_;
            column_ = 1;
        } else {
            value << source_[pos_];
            ++pos_;
            ++column_;
        }
    }

    token.value = value.str();
    return token;
}

GraphQLLexer::Token GraphQLLexer::nextToken() {
    skipWhitespace();

    if (pos_ >= source_.size()) {
        return {TokenType::EOF_TOKEN, "", line_, column_};
    }

    char c = source_[pos_];

    // Check for block string
    if (c == '"' && pos_ + 2 < source_.size() &&
        source_[pos_ + 1] == '"' && source_[pos_ + 2] == '"') {
        return readBlockString();
    }

    // Check for string
    if (c == '"') {
        return readString();
    }

    // Check for number
    if (c == '-' || (c >= '0' && c <= '9')) {
        return readNumber();
    }

    // Check for name
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
        return readName();
    }

    // Punctuators
    Token token;
    token.type = TokenType::Punctuator;
    token.line = line_;
    token.column = column_;

    // Multi-character punctuators
    if (c == '.' && pos_ + 2 < source_.size() &&
        source_[pos_ + 1] == '.' && source_[pos_ + 2] == '.') {
        token.value = "...";
        pos_ += 3;
        column_ += 3;
        return token;
    }

    // Single character punctuators
    token.value = std::string(1, c);
    ++pos_;
    ++column_;
    return token;
}

GraphQLLexer::Token GraphQLLexer::peekToken() {
    size_t savedPos = pos_;
    uint32_t savedLine = line_;
    uint32_t savedColumn = column_;

    Token token = nextToken();

    pos_ = savedPos;
    line_ = savedLine;
    column_ = savedColumn;

    return token;
}

// ============================================================================
// GraphQL Parser Implementation
// ============================================================================

GraphQLParser::GraphQLParser(const std::string& source) : lexer_(source) {
    advance();
}

void GraphQLParser::advance() {
    currentToken_ = lexer_.nextToken();
}

bool GraphQLParser::expect(const std::string& value) {
    if (currentToken_.value == value) {
        advance();
        return true;
    }
    return false;
}

bool GraphQLParser::match(const std::string& value) {
    return currentToken_.value == value;
}

QueryTranslator::ParseResult GraphQLParser::parse() {
    QueryTranslator::ParseResult result;
    result.success = false;

    auto document = parseDocument();
    if (document) {
        result.success = true;
        result.document = std::move(document);
    } else {
        result.errors.push_back("Failed to parse GraphQL document");
    }

    return result;
}

std::optional<GraphQLDocument> GraphQLParser::parseDocument() {
    GraphQLDocument doc;

    // Parse operation type (query/mutation/subscription)
    if (match("query")) {
        doc.operationType = GraphQLOperationType::Query;
        advance();
    } else if (match("mutation")) {
        doc.operationType = GraphQLOperationType::Mutation;
        advance();
    } else if (match("subscription")) {
        doc.operationType = GraphQLOperationType::Subscription;
        advance();
    } else if (match("{")) {
        // Anonymous query
        doc.operationType = GraphQLOperationType::Query;
    } else {
        return std::nullopt;
    }

    // Optional operation name
    if (currentToken_.type == GraphQLLexer::TokenType::Name && !match("{")) {
        doc.operationName = currentToken_.value;
        advance();
    }

    // Optional variable definitions
    if (match("(")) {
        auto vars = parseVariableDefinitions();
        if (vars) {
            doc.variables = std::move(*vars);
        }
    }

    // Selection set
    auto selections = parseSelectionSet();
    if (selections) {
        doc.selections = std::move(*selections);
    } else {
        return std::nullopt;
    }

    return doc;
}

std::optional<std::vector<GraphQLVariable>> GraphQLParser::parseVariableDefinitions() {
    std::vector<GraphQLVariable> vars;

    if (!expect("(")) {
        return std::nullopt;
    }

    while (!match(")") && currentToken_.type != GraphQLLexer::TokenType::EOF_TOKEN) {
        GraphQLVariable var;

        // Variable name starts with $
        if (!expect("$")) {
            return std::nullopt;
        }

        if (currentToken_.type != GraphQLLexer::TokenType::Name) {
            return std::nullopt;
        }
        var.name = currentToken_.value;
        advance();

        if (!expect(":")) {
            return std::nullopt;
        }

        // Type name
        std::string typeName;
        if (match("[")) {
            typeName = "[";
            advance();
        }

        if (currentToken_.type != GraphQLLexer::TokenType::Name) {
            return std::nullopt;
        }
        typeName += currentToken_.value;
        advance();

        if (match("!")) {
            typeName += "!";
            advance();
        }

        if (match("]")) {
            typeName += "]";
            advance();
            if (match("!")) {
                typeName += "!";
                advance();
            }
        }

        var.typeName = typeName;

        // Optional default value
        if (match("=")) {
            advance();
            // Read value (simplified)
            if (currentToken_.type == GraphQLLexer::TokenType::StringValue ||
                currentToken_.type == GraphQLLexer::TokenType::IntValue ||
                currentToken_.type == GraphQLLexer::TokenType::FloatValue ||
                currentToken_.type == GraphQLLexer::TokenType::Name) {
                var.defaultValue = currentToken_.value;
                advance();
            }
        }

        vars.push_back(var);
    }

    if (!expect(")")) {
        return std::nullopt;
    }

    return vars;
}

std::optional<std::vector<std::shared_ptr<GraphQLSelection>>> GraphQLParser::parseSelectionSet() {
    std::vector<std::shared_ptr<GraphQLSelection>> selections;

    if (!expect("{")) {
        return std::nullopt;
    }

    while (!match("}") && currentToken_.type != GraphQLLexer::TokenType::EOF_TOKEN) {
        if (match("...")) {
            // Fragment spread or inline fragment
            advance();

            if (match("on")) {
                // Inline fragment
                advance();
                GraphQLInlineFragment fragment;

                if (currentToken_.type == GraphQLLexer::TokenType::Name) {
                    fragment.typeCondition = currentToken_.value;
                    advance();
                }

                auto subSelections = parseSelectionSet();
                if (subSelections) {
                    fragment.selections = std::move(*subSelections);
                }

                auto selection = std::make_shared<GraphQLSelection>();
                selection->selection = fragment;
                selections.push_back(selection);
            } else if (currentToken_.type == GraphQLLexer::TokenType::Name) {
                // Fragment spread
                GraphQLFragmentSpread spread;
                spread.fragmentName = currentToken_.value;
                advance();

                auto selection = std::make_shared<GraphQLSelection>();
                selection->selection = spread;
                selections.push_back(selection);
            }
        } else {
            // Field
            auto field = parseField();
            if (field) {
                auto selection = std::make_shared<GraphQLSelection>();
                selection->selection = std::move(*field);
                selections.push_back(selection);
            }
        }
    }

    if (!expect("}")) {
        return std::nullopt;
    }

    return selections;
}

std::optional<GraphQLFieldSelection> GraphQLParser::parseField() {
    GraphQLFieldSelection field;

    // Check for alias
    if (currentToken_.type == GraphQLLexer::TokenType::Name) {
        std::string name = currentToken_.value;
        advance();

        if (match(":")) {
            // This was an alias
            field.alias = name;
            advance();

            if (currentToken_.type != GraphQLLexer::TokenType::Name) {
                return std::nullopt;
            }
            field.name = currentToken_.value;
            advance();
        } else {
            field.name = name;
        }
    } else {
        return std::nullopt;
    }

    // Optional arguments
    if (match("(")) {
        auto args = parseArguments();
        if (args) {
            field.arguments = std::move(*args);
        }
    }

    // Optional selection set
    if (match("{")) {
        auto selections = parseSelectionSet();
        if (selections) {
            field.selections = std::move(*selections);
        }
    }

    return field;
}

std::optional<std::unordered_map<std::string, std::string>> GraphQLParser::parseArguments() {
    std::unordered_map<std::string, std::string> args;

    if (!expect("(")) {
        return std::nullopt;
    }

    while (!match(")") && currentToken_.type != GraphQLLexer::TokenType::EOF_TOKEN) {
        if (currentToken_.type != GraphQLLexer::TokenType::Name) {
            return std::nullopt;
        }
        std::string argName = currentToken_.value;
        advance();

        if (!expect(":")) {
            return std::nullopt;
        }

        // Parse value (simplified - handles basic types)
        std::string value;
        if (currentToken_.type == GraphQLLexer::TokenType::StringValue) {
            value = "\"" + currentToken_.value + "\"";
        } else if (currentToken_.type == GraphQLLexer::TokenType::IntValue ||
                   currentToken_.type == GraphQLLexer::TokenType::FloatValue ||
                   currentToken_.type == GraphQLLexer::TokenType::Name) {
            value = currentToken_.value;
        } else if (match("$")) {
            advance();
            if (currentToken_.type == GraphQLLexer::TokenType::Name) {
                value = "$" + currentToken_.value;
            }
        }
        advance();

        args[argName] = value;
    }

    if (!expect(")")) {
        return std::nullopt;
    }

    return args;
}

// ============================================================================
// Query Translator Implementation
// ============================================================================

QueryTranslator::QueryTranslator(const SchemaRegistry& schema, const TranslationOptions& options)
    : schema_(schema), options_(options) {}

QueryTranslator::ParseResult QueryTranslator::parse(const std::string& query) {
    GraphQLParser parser(query);
    return parser.parse();
}

QueryTranslator::TranslationResult QueryTranslator::translate(
    const GraphQLDocument& document,
    const std::unordered_map<std::string, std::string>& variables) {

    TranslationResult result;
    result.success = true;

    // Validate document against schema
    auto validation = validate(document);
    if (!validation.valid) {
        result.success = false;
        result.errors = validation.errors;
        return result;
    }

    // Translate each top-level selection
    for (const auto& selection : document.selections) {
        if (auto* field = std::get_if<GraphQLFieldSelection>(&selection->selection)) {
            auto cypher = translateSelection(*field, "", 0);
            if (!cypher.empty()) {
                CypherQuery query;
                query.query = cypher;

                // Apply variable substitution
                for (const auto& [name, value] : variables) {
                    std::string placeholder = "$" + name;
                    size_t pos = 0;
                    while ((pos = query.query.find(placeholder, pos)) != std::string::npos) {
                        query.query.replace(pos, placeholder.length(), value);
                        pos += value.length();
                    }
                }

                // Check for special operations
                if (field->name == "semanticSearch" || field->isSemanticSearch) {
                    query.requiresVectorDB = true;
                    query.vectorSearchQuery = translateSemanticSearch(*field);
                }

                if (field->name == "reason" || field->isReasoningQuery) {
                    query.requiresAGIProcessing = true;
                }

                if (options_.enableOptimizations) {
                    optimizeQuery(query);
                }

                result.queries.push_back(query);
            }
        }
    }

    return result;
}

QueryTranslator::TranslationResult QueryTranslator::translateQuery(
    const std::string& graphqlQuery,
    const std::unordered_map<std::string, std::string>& variables) {

    auto parseResult = parse(graphqlQuery);
    if (!parseResult.success) {
        TranslationResult result;
        result.success = false;
        result.errors = parseResult.errors;
        return result;
    }

    return translate(*parseResult.document, variables);
}

QueryTranslator::ValidationResult QueryTranslator::validate(const GraphQLDocument& document) {
    ValidationResult result;
    result.valid = true;

    // Check operation type is supported
    if (document.operationType == GraphQLOperationType::Subscription) {
        result.warnings.push_back("Subscriptions are not fully supported yet");
    }

    // Validate each selection
    for (const auto& selection : document.selections) {
        if (auto* field = std::get_if<GraphQLFieldSelection>(&selection->selection)) {
            // Get the query type
            const auto* queryType = schema_.getType(SchemaRegistry::QUERY_TYPE);
            if (!queryType) {
                result.errors.push_back("Schema has no Query type");
                result.valid = false;
                continue;
            }

            // Check if field exists on Query type
            bool fieldFound = false;
            for (const auto& queryField : queryType->fields) {
                if (queryField.name == field->name) {
                    fieldFound = true;
                    break;
                }
            }

            if (!fieldFound) {
                result.errors.push_back("Unknown field '" + field->name + "' on Query");
                result.valid = false;
            }
        }
    }

    return result;
}

uint32_t QueryTranslator::calculateComplexity(const GraphQLDocument& document) {
    uint32_t complexity = 0;

    std::function<void(const std::vector<std::shared_ptr<GraphQLSelection>>&, uint32_t)> calculate;
    calculate = [&](const std::vector<std::shared_ptr<GraphQLSelection>>& selections, uint32_t depth) {
        for (const auto& selection : selections) {
            if (auto* field = std::get_if<GraphQLFieldSelection>(&selection->selection)) {
                complexity += (1 + depth);  // Base complexity increases with depth
                calculate(field->selections, depth + 1);
            } else if (auto* fragment = std::get_if<GraphQLInlineFragment>(&selection->selection)) {
                complexity += 1;
                calculate(fragment->selections, depth);
            }
        }
    };

    calculate(document.selections, 0);
    return complexity;
}

bool QueryTranslator::checkDepth(const GraphQLDocument& document, uint32_t maxDepth) {
    std::function<uint32_t(const std::vector<std::shared_ptr<GraphQLSelection>>&)> getDepth;
    getDepth = [&](const std::vector<std::shared_ptr<GraphQLSelection>>& selections) -> uint32_t {
        uint32_t maxChildDepth = 0;
        for (const auto& selection : selections) {
            if (auto* field = std::get_if<GraphQLFieldSelection>(&selection->selection)) {
                if (!field->selections.empty()) {
                    maxChildDepth = std::max(maxChildDepth, 1 + getDepth(field->selections));
                }
            } else if (auto* fragment = std::get_if<GraphQLInlineFragment>(&selection->selection)) {
                maxChildDepth = std::max(maxChildDepth, getDepth(fragment->selections));
            }
        }
        return maxChildDepth;
    };

    return getDepth(document.selections) <= maxDepth;
}

std::string QueryTranslator::translateSelection(
    const GraphQLFieldSelection& field,
    const std::string& parentAlias,
    uint32_t depth) {

    std::ostringstream cypher;

    // Get field type from schema
    const auto* queryType = schema_.getType(SchemaRegistry::QUERY_TYPE);
    if (!queryType) return "";

    // Find the field definition
    const GraphQLFieldDefinition* fieldDef = nullptr;
    for (const auto& f : queryType->fields) {
        if (f.name == field.name) {
            fieldDef = &f;
            break;
        }
    }

    if (!fieldDef) {
        // Check if it's a special AGI operation
        if (field.name == "semanticSearch") {
            return translateSemanticSearch(field);
        }
        if (field.name == "reason") {
            return translateReasoningQuery(field);
        }
        return "";
    }

    // Determine the Kuzu table name
    std::string typeName = fieldDef->typeName;
    // Remove list brackets and non-null markers
    while (!typeName.empty() && (typeName.front() == '[' || typeName.back() == ']' ||
                                   typeName.back() == '!')) {
        if (typeName.front() == '[') typeName = typeName.substr(1);
        if (!typeName.empty() && typeName.back() == ']') typeName.pop_back();
        if (!typeName.empty() && typeName.back() == '!') typeName.pop_back();
    }

    const auto* nodeType = schema_.getType(typeName);
    if (!nodeType || !nodeType->isNodeType) return "";

    std::string alias = field.alias.value_or(field.name);
    std::string nodeAlias = alias + "_n";

    // Build MATCH clause
    cypher << "MATCH (" << nodeAlias << ":" << nodeType->kuzuTableName << ")";

    // Add WHERE clause from arguments
    std::string whereClause = translateArguments(field.arguments, nodeAlias);
    if (!whereClause.empty()) {
        cypher << " WHERE " << whereClause;
    }

    // Build RETURN clause
    cypher << " RETURN ";

    std::vector<std::string> returnFields;
    if (field.selections.empty()) {
        // Return all properties
        returnFields.push_back(nodeAlias);
    } else {
        for (const auto& sel : field.selections) {
            if (auto* subField = std::get_if<GraphQLFieldSelection>(&sel->selection)) {
                returnFields.push_back(nodeAlias + "." + subField->name + " AS " +
                                        (subField->alias.value_or(subField->name)));
            }
        }
    }

    for (size_t i = 0; i < returnFields.size(); ++i) {
        if (i > 0) cypher << ", ";
        cypher << returnFields[i];
    }

    // Add pagination
    std::string pagination = translatePagination(field.arguments);
    if (!pagination.empty()) {
        cypher << " " << pagination;
    }

    return cypher.str();
}

std::string QueryTranslator::translateArguments(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& alias) {

    std::vector<std::string> conditions;

    for (const auto& [name, value] : args) {
        if (name == "first" || name == "after" || name == "before" ||
            name == "last" || name == "orderBy" || name == "offset") {
            continue;  // Handled by pagination
        }

        if (name == "id") {
            conditions.push_back(alias + ".id = " + value);
        } else if (name == "filter") {
            // TODO: Parse filter JSON
            continue;
        } else {
            // Direct property match
            conditions.push_back(alias + "." + name + " = " + value);
        }
    }

    std::ostringstream result;
    for (size_t i = 0; i < conditions.size(); ++i) {
        if (i > 0) result << " AND ";
        result << conditions[i];
    }

    return result.str();
}

std::string QueryTranslator::translatePagination(
    const std::unordered_map<std::string, std::string>& args) {

    std::ostringstream result;

    auto orderByIt = args.find("orderBy");
    if (orderByIt != args.end()) {
        result << translateOrderBy(orderByIt->second) << " ";
    }

    auto offsetIt = args.find("offset");
    auto afterIt = args.find("after");
    if (offsetIt != args.end()) {
        result << "SKIP " << offsetIt->second << " ";
    } else if (afterIt != args.end()) {
        // Decode cursor and use as offset
        // TODO: Implement cursor decoding
    }

    auto firstIt = args.find("first");
    auto lastIt = args.find("last");
    if (firstIt != args.end()) {
        result << "LIMIT " << firstIt->second;
    } else if (lastIt != args.end()) {
        result << "LIMIT " << lastIt->second;
    } else {
        result << "LIMIT " << options_.defaultLimit;
    }

    return result.str();
}

std::string QueryTranslator::translateOrderBy(const std::string& orderByArg) {
    // Parse orderBy argument (format: "field_ASC" or "field_DESC")
    std::string field = orderByArg;
    std::string direction = "ASC";

    size_t underscorePos = orderByArg.rfind('_');
    if (underscorePos != std::string::npos) {
        std::string suffix = orderByArg.substr(underscorePos + 1);
        if (suffix == "ASC" || suffix == "DESC") {
            field = orderByArg.substr(0, underscorePos);
            direction = suffix;
        }
    }

    return "ORDER BY n." + field + " " + direction;
}

std::string QueryTranslator::translateSemanticSearch(const GraphQLFieldSelection& field) {
    std::ostringstream cypher;

    // Extract search parameters from arguments
    std::string query;
    std::string table = "*";
    uint32_t topK = 10;

    auto queryIt = field.arguments.find("query");
    if (queryIt != field.arguments.end()) {
        query = queryIt->second;
        // Remove quotes if present
        if (query.size() >= 2 && query.front() == '"' && query.back() == '"') {
            query = query.substr(1, query.size() - 2);
        }
    }

    auto tableIt = field.arguments.find("table");
    if (tableIt != field.arguments.end()) {
        table = tableIt->second;
    }

    auto topKIt = field.arguments.find("topK");
    if (topKIt != field.arguments.end()) {
        topK = std::stoul(topKIt->second);
    }

    // Generate semantic search Cypher using vector extension
    cypher << "CALL SEMANTIC_SEARCH('" << query << "', '" << table << "', " << topK << ") "
           << "YIELD entity_id, score, entity_type, content "
           << "RETURN entity_id, score, entity_type, content";

    return cypher.str();
}

std::string QueryTranslator::translateReasoningQuery(const GraphQLFieldSelection& field) {
    std::ostringstream cypher;

    std::string question;
    auto questionIt = field.arguments.find("question");
    if (questionIt != field.arguments.end()) {
        question = questionIt->second;
    }

    // This returns a placeholder - actual reasoning happens in the AGI engine
    cypher << "CALL AGI_REASON('" << question << "') "
           << "YIELD answer, confidence, steps "
           << "RETURN answer, confidence, steps";

    return cypher.str();
}

void QueryTranslator::optimizeQuery(CypherQuery& query) {
    // Apply filter pushdown
    pushDownFilters(query);

    // Optimize joins
    optimizeJoins(query);
}

void QueryTranslator::pushDownFilters(CypherQuery& query) {
    // Simple filter pushdown - move WHERE conditions closer to MATCH
    // This is a placeholder for more sophisticated optimization
}

void QueryTranslator::optimizeJoins(CypherQuery& query) {
    // Reorder joins for better performance
    // This is a placeholder for join optimization
}

} // namespace graphql_agi
} // namespace kuzu
