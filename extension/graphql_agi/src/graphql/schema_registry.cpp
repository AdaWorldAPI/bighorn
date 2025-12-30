/**
 * GraphQL Schema Registry Implementation
 *
 * Manages GraphQL type definitions and schema building from Kuzu catalog.
 */

#include "graphql/schema_registry.h"
#include "main/client_context.h"
#include "catalog/catalog.h"

#include <sstream>
#include <algorithm>

namespace kuzu {
namespace graphql_agi {

void SchemaRegistry::registerType(const GraphQLTypeDefinition& typeDef) {
    types_[typeDef.name] = typeDef;
}

void SchemaRegistry::registerDirective(const GraphQLDirective& directive) {
    directives_[directive.name] = directive;
}

const GraphQLTypeDefinition* SchemaRegistry::getType(const std::string& name) const {
    auto it = types_.find(name);
    return it != types_.end() ? &it->second : nullptr;
}

std::vector<const GraphQLTypeDefinition*> SchemaRegistry::getAllTypes() const {
    std::vector<const GraphQLTypeDefinition*> result;
    for (const auto& [name, type] : types_) {
        result.push_back(&type);
    }
    return result;
}

const GraphQLDirective* SchemaRegistry::getDirective(const std::string& name) const {
    auto it = directives_.find(name);
    return it != directives_.end() ? &it->second : nullptr;
}

void SchemaRegistry::buildFromKuzuCatalog(main::ClientContext* context) {
    // Clear existing types
    types_.clear();
    directives_.clear();

    // Add built-in types first
    addBuiltInScalars();
    addBuiltInDirectives();
    addRelayTypes();

    // Get catalog
    auto catalog = context->getCatalog();

    // Create Query root type
    GraphQLTypeDefinition queryType;
    queryType.name = QUERY_TYPE;
    queryType.kind = GraphQLTypeKind::Object;
    queryType.description = "Root query type for GraphQL AGI";

    // Create Mutation root type
    GraphQLTypeDefinition mutationType;
    mutationType.name = MUTATION_TYPE;
    mutationType.kind = GraphQLTypeKind::Object;
    mutationType.description = "Root mutation type for GraphQL AGI";

    // Iterate through all node tables
    auto nodeTableEntries = catalog->getNodeTableEntries(context->getTransaction());
    for (const auto& entry : nodeTableEntries) {
        std::string tableName = entry->getName();

        // Create GraphQL type for node table
        GraphQLTypeDefinition nodeType;
        nodeType.name = tableName;
        nodeType.kind = GraphQLTypeKind::Object;
        nodeType.description = "Node type: " + tableName;
        nodeType.kuzuTableName = tableName;
        nodeType.isNodeType = true;
        nodeType.interfaces.push_back("Node");  // Relay Node interface

        // Add ID field (required for Relay)
        GraphQLFieldDefinition idField;
        idField.name = "id";
        idField.typeName = "ID!";
        idField.description = "Unique identifier";
        nodeType.fields.push_back(idField);

        // Add properties as fields
        auto properties = entry->getProperties();
        for (const auto& prop : properties) {
            GraphQLFieldDefinition field;
            field.name = prop.getName();
            field.typeName = kuzuTypeToGraphQL(prop.getDataType().toString());
            field.description = "Property: " + prop.getName();

            // Check if this might be an embeddable field
            auto typeStr = prop.getDataType().toString();
            if (typeStr.find("STRING") != std::string::npos ||
                typeStr.find("VARCHAR") != std::string::npos) {
                field.isEmbeddable = true;
            }

            nodeType.fields.push_back(field);
        }

        // Check for vector/embedding support
        for (const auto& prop : properties) {
            auto typeStr = prop.getDataType().toString();
            if (typeStr.find("FLOAT") != std::string::npos &&
                typeStr.find("[]") != std::string::npos) {
                nodeType.supportsVectorSearch = true;
                nodeType.embeddingField = prop.getName();
                break;
            }
        }

        types_[nodeType.name] = nodeType;

        // Add query field for this type
        GraphQLFieldDefinition queryField;
        queryField.name = toLowerCamelCase(tableName);
        queryField.typeName = tableName;
        queryField.description = "Get a single " + tableName + " by ID";

        GraphQLArgument idArg;
        idArg.name = "id";
        idArg.typeName = "ID!";
        idArg.required = true;
        queryField.arguments.push_back(idArg);

        queryType.fields.push_back(queryField);

        // Add list query field
        GraphQLFieldDefinition listField;
        listField.name = toLowerCamelCase(tableName) + "s";
        listField.typeName = "[" + tableName + "!]!";
        listField.description = "Get all " + tableName + " entities";

        // Add pagination arguments
        GraphQLArgument firstArg, afterArg, filterArg, orderByArg;
        firstArg.name = "first";
        firstArg.typeName = "Int";
        firstArg.defaultValue = "10";

        afterArg.name = "after";
        afterArg.typeName = "String";

        filterArg.name = "filter";
        filterArg.typeName = tableName + "Filter";

        orderByArg.name = "orderBy";
        orderByArg.typeName = tableName + "OrderBy";

        listField.arguments = {firstArg, afterArg, filterArg, orderByArg};
        queryType.fields.push_back(listField);

        // Add Connection type for Relay pagination
        GraphQLTypeDefinition connectionType;
        connectionType.name = tableName + "Connection";
        connectionType.kind = GraphQLTypeKind::Connection;
        connectionType.description = "Connection type for " + tableName;

        GraphQLFieldDefinition edgesField;
        edgesField.name = "edges";
        edgesField.typeName = "[" + tableName + "Edge!]!";

        GraphQLFieldDefinition pageInfoField;
        pageInfoField.name = "pageInfo";
        pageInfoField.typeName = "PageInfo!";

        GraphQLFieldDefinition totalCountField;
        totalCountField.name = "totalCount";
        totalCountField.typeName = "Int!";

        connectionType.fields = {edgesField, pageInfoField, totalCountField};
        types_[connectionType.name] = connectionType;

        // Add Edge type
        GraphQLTypeDefinition edgeType;
        edgeType.name = tableName + "Edge";
        edgeType.kind = GraphQLTypeKind::Edge;

        GraphQLFieldDefinition nodeField;
        nodeField.name = "node";
        nodeField.typeName = tableName + "!";

        GraphQLFieldDefinition cursorField;
        cursorField.name = "cursor";
        cursorField.typeName = "String!";

        edgeType.fields = {nodeField, cursorField};
        types_[edgeType.name] = edgeType;

        // Add Filter input type
        GraphQLTypeDefinition filterType;
        filterType.name = tableName + "Filter";
        filterType.kind = GraphQLTypeKind::InputObject;
        filterType.description = "Filter input for " + tableName;

        for (const auto& prop : properties) {
            // Add comparison fields
            GraphQLFieldDefinition eqField;
            eqField.name = prop.getName();
            eqField.typeName = kuzuTypeToGraphQL(prop.getDataType().toString());

            GraphQLFieldDefinition neField;
            neField.name = prop.getName() + "_ne";
            neField.typeName = kuzuTypeToGraphQL(prop.getDataType().toString());

            filterType.fields.push_back(eqField);
            filterType.fields.push_back(neField);
        }

        // Add AND/OR/NOT for complex filters
        GraphQLFieldDefinition andField, orField, notField;
        andField.name = "AND";
        andField.typeName = "[" + tableName + "Filter!]";
        orField.name = "OR";
        orField.typeName = "[" + tableName + "Filter!]";
        notField.name = "NOT";
        notField.typeName = tableName + "Filter";

        filterType.fields.push_back(andField);
        filterType.fields.push_back(orField);
        filterType.fields.push_back(notField);

        types_[filterType.name] = filterType;

        // Add mutation for creating entities
        GraphQLFieldDefinition createMutation;
        createMutation.name = "create" + tableName;
        createMutation.typeName = tableName + "!";
        createMutation.description = "Create a new " + tableName;

        GraphQLArgument inputArg;
        inputArg.name = "input";
        inputArg.typeName = "Create" + tableName + "Input!";
        inputArg.required = true;
        createMutation.arguments.push_back(inputArg);

        mutationType.fields.push_back(createMutation);
    }

    // Iterate through relationship tables
    auto relTableEntries = catalog->getRelTableEntries(context->getTransaction());
    for (const auto& entry : relTableEntries) {
        std::string tableName = entry->getName();

        // Create GraphQL type for relationship
        GraphQLTypeDefinition relType;
        relType.name = tableName;
        relType.kind = GraphQLTypeKind::Object;
        relType.description = "Relationship type: " + tableName;
        relType.kuzuTableName = tableName;
        relType.isRelationType = true;

        // Add properties
        auto properties = entry->getProperties();
        for (const auto& prop : properties) {
            GraphQLFieldDefinition field;
            field.name = prop.getName();
            field.typeName = kuzuTypeToGraphQL(prop.getDataType().toString());
            relType.fields.push_back(field);
        }

        types_[relType.name] = relType;

        // Add relationship fields to connected node types
        auto srcTableId = entry->getSrcTableID();
        auto dstTableId = entry->getDstTableID();

        // Find source and destination table names
        for (auto& [name, type] : types_) {
            if (type.isNodeType) {
                // Add outgoing relationship field
                GraphQLFieldDefinition relField;
                relField.name = toLowerCamelCase(tableName);
                relField.typeName = "[" + tableName + "!]!";
                relField.description = "Outgoing " + tableName + " relationships";

                // Add filter and pagination arguments
                GraphQLArgument firstArg;
                firstArg.name = "first";
                firstArg.typeName = "Int";
                firstArg.defaultValue = "10";
                relField.arguments.push_back(firstArg);

                type.fields.push_back(relField);
            }
        }
    }

    // Register root types
    types_[queryType.name] = queryType;
    types_[mutationType.name] = mutationType;
}

std::string SchemaRegistry::toSDL() const {
    std::ostringstream sdl;

    // Add schema definition
    sdl << "schema {\n";
    sdl << "  query: Query\n";
    sdl << "  mutation: Mutation\n";
    sdl << "}\n\n";

    // Add directives
    for (const auto& [name, directive] : directives_) {
        sdl << "directive @" << directive.name;
        if (!directive.arguments.empty()) {
            sdl << "(";
            for (size_t i = 0; i < directive.arguments.size(); ++i) {
                if (i > 0) sdl << ", ";
                const auto& arg = directive.arguments[i];
                sdl << arg.name << ": " << arg.typeName;
                if (arg.defaultValue) {
                    sdl << " = " << *arg.defaultValue;
                }
            }
            sdl << ")";
        }
        sdl << " on ";
        for (size_t i = 0; i < directive.locations.size(); ++i) {
            if (i > 0) sdl << " | ";
            sdl << directive.locations[i];
        }
        sdl << "\n\n";
    }

    // Add types
    for (const auto& [name, type] : types_) {
        if (type.kind == GraphQLTypeKind::Scalar) {
            sdl << "scalar " << type.name << "\n\n";
            continue;
        }

        if (type.kind == GraphQLTypeKind::Enum) {
            sdl << "enum " << type.name << " {\n";
            for (const auto& field : type.fields) {
                sdl << "  " << field.name << "\n";
            }
            sdl << "}\n\n";
            continue;
        }

        if (type.kind == GraphQLTypeKind::Interface) {
            sdl << "interface " << type.name << " {\n";
        } else if (type.kind == GraphQLTypeKind::InputObject) {
            sdl << "input " << type.name << " {\n";
        } else {
            sdl << "type " << type.name;
            if (!type.interfaces.empty()) {
                sdl << " implements ";
                for (size_t i = 0; i < type.interfaces.size(); ++i) {
                    if (i > 0) sdl << " & ";
                    sdl << type.interfaces[i];
                }
            }
            sdl << " {\n";
        }

        for (const auto& field : type.fields) {
            sdl << "  " << field.name;
            if (!field.arguments.empty()) {
                sdl << "(";
                for (size_t i = 0; i < field.arguments.size(); ++i) {
                    if (i > 0) sdl << ", ";
                    const auto& arg = field.arguments[i];
                    sdl << arg.name << ": " << arg.typeName;
                    if (arg.defaultValue) {
                        sdl << " = " << *arg.defaultValue;
                    }
                }
                sdl << ")";
            }
            sdl << ": " << field.typeName;
            if (field.deprecated) {
                sdl << " @deprecated";
                if (!field.deprecationReason.empty()) {
                    sdl << "(reason: \"" << field.deprecationReason << "\")";
                }
            }
            sdl << "\n";
        }

        sdl << "}\n\n";
    }

    return sdl.str();
}

SchemaRegistry::ValidationResult SchemaRegistry::validate() const {
    ValidationResult result;
    result.valid = true;

    // Check that Query type exists
    if (types_.find(QUERY_TYPE) == types_.end()) {
        result.errors.push_back("Missing Query type");
        result.valid = false;
    }

    // Validate all type references
    for (const auto& [name, type] : types_) {
        for (const auto& field : type.fields) {
            std::string typeName = field.typeName;

            // Remove list brackets and non-null markers
            while (!typeName.empty() && (typeName.front() == '[' || typeName.back() == ']' ||
                                           typeName.back() == '!')) {
                if (typeName.front() == '[') typeName = typeName.substr(1);
                if (!typeName.empty() && typeName.back() == ']') typeName.pop_back();
                if (!typeName.empty() && typeName.back() == '!') typeName.pop_back();
            }

            // Check if type exists
            if (!typeName.empty() && types_.find(typeName) == types_.end()) {
                // Check if it's a built-in scalar
                if (typeName != "ID" && typeName != "String" && typeName != "Int" &&
                    typeName != "Float" && typeName != "Boolean" && typeName != "DateTime" &&
                    typeName != "JSON") {
                    result.warnings.push_back("Unknown type '" + typeName + "' referenced in " +
                                               name + "." + field.name);
                }
            }
        }

        // Validate interfaces
        for (const auto& iface : type.interfaces) {
            auto ifaceIt = types_.find(iface);
            if (ifaceIt == types_.end()) {
                result.errors.push_back("Type " + name + " implements unknown interface " + iface);
                result.valid = false;
            } else if (ifaceIt->second.kind != GraphQLTypeKind::Interface) {
                result.errors.push_back("Type " + name + " implements " + iface +
                                         " which is not an interface");
                result.valid = false;
            }
        }
    }

    return result;
}

void SchemaRegistry::addAGITypes() {
    // Add SemanticSearchInput
    GraphQLTypeDefinition semanticSearchInput;
    semanticSearchInput.name = "SemanticSearchInput";
    semanticSearchInput.kind = GraphQLTypeKind::InputObject;
    semanticSearchInput.description = "Input for semantic search queries";

    semanticSearchInput.fields = {
        {"query", "String!", "The search query text", {}, false, "", true, false, ""},
        {"model", "String", "Embedding model to use", {}, false, "", false, false, ""},
        {"topK", "Int", "Number of results to return", {}, false, "", false, false, ""},
        {"threshold", "Float", "Similarity threshold (0-1)", {}, false, "", false, false, ""},
        {"filters", "[String!]", "Additional filters", {}, false, "", false, false, ""}
    };

    types_[semanticSearchInput.name] = semanticSearchInput;

    // Add ReasoningRequest
    GraphQLTypeDefinition reasoningRequest;
    reasoningRequest.name = "ReasoningRequest";
    reasoningRequest.kind = GraphQLTypeKind::InputObject;
    reasoningRequest.description = "Input for AGI reasoning queries";

    reasoningRequest.fields = {
        {"question", "String!", "The question to reason about", {}, false, "", false, false, ""},
        {"context", "[String!]", "Additional context", {}, false, "", false, false, ""},
        {"strategy", "ReasoningStrategy", "Reasoning strategy to use", {}, false, "", false, false, ""},
        {"maxSteps", "Int", "Maximum reasoning steps", {}, false, "", false, false, ""}
    };

    types_[reasoningRequest.name] = reasoningRequest;

    // Add ReasoningStrategy enum
    GraphQLTypeDefinition reasoningStrategy;
    reasoningStrategy.name = "ReasoningStrategy";
    reasoningStrategy.kind = GraphQLTypeKind::Enum;
    reasoningStrategy.description = "Available reasoning strategies";

    reasoningStrategy.fields = {
        {"CHAIN_OF_THOUGHT", "", "", {}, false, "", false, false, ""},
        {"TREE_OF_THOUGHTS", "", "", {}, false, "", false, false, ""},
        {"GRAPH_OF_THOUGHTS", "", "", {}, false, "", false, false, ""},
        {"REACT", "", "", {}, false, "", false, false, ""},
        {"SELF_CONSISTENCY", "", "", {}, false, "", false, false, ""},
        {"REFLEXION", "", "", {}, false, "", false, false, ""},
        {"PLAN_AND_EXECUTE", "", "", {}, false, "", false, false, ""}
    };

    types_[reasoningStrategy.name] = reasoningStrategy;

    // Add ReasoningResult
    GraphQLTypeDefinition reasoningResult;
    reasoningResult.name = "ReasoningResult";
    reasoningResult.kind = GraphQLTypeKind::Object;
    reasoningResult.description = "Result of AGI reasoning";

    reasoningResult.fields = {
        {"success", "Boolean!", "Whether reasoning succeeded", {}, false, "", false, false, ""},
        {"answer", "String!", "The reasoned answer", {}, false, "", false, false, ""},
        {"confidence", "Float!", "Confidence score (0-1)", {}, false, "", false, false, ""},
        {"steps", "[ReasoningStep!]!", "Reasoning steps taken", {}, false, "", false, false, ""},
        {"relatedEntities", "[String!]!", "Related entities discovered", {}, false, "", false, false, ""},
        {"usedSources", "[String!]!", "Sources used in reasoning", {}, false, "", false, false, ""}
    };

    types_[reasoningResult.name] = reasoningResult;

    // Add ReasoningStep
    GraphQLTypeDefinition reasoningStep;
    reasoningStep.name = "ReasoningStep";
    reasoningStep.kind = GraphQLTypeKind::Object;
    reasoningStep.description = "A single step in the reasoning process";

    reasoningStep.fields = {
        {"stepNumber", "Int!", "Step number in sequence", {}, false, "", false, false, ""},
        {"thought", "String!", "The thought/reasoning", {}, false, "", false, false, ""},
        {"action", "String!", "Action taken", {}, false, "", false, false, ""},
        {"observation", "String", "Observation from action", {}, false, "", false, false, ""},
        {"confidence", "Float!", "Confidence for this step", {}, false, "", false, false, ""}
    };

    types_[reasoningStep.name] = reasoningStep;

    // Add Query fields for AGI
    auto queryIt = types_.find(QUERY_TYPE);
    if (queryIt != types_.end()) {
        GraphQLFieldDefinition semanticSearchField;
        semanticSearchField.name = "semanticSearch";
        semanticSearchField.typeName = "[SemanticSearchResult!]!";
        semanticSearchField.description = "Perform semantic search across the knowledge graph";
        semanticSearchField.isSemanticSearch = true;

        GraphQLArgument inputArg;
        inputArg.name = "input";
        inputArg.typeName = "SemanticSearchInput!";
        inputArg.required = true;
        semanticSearchField.arguments.push_back(inputArg);

        queryIt->second.fields.push_back(semanticSearchField);

        GraphQLFieldDefinition reasonField;
        reasonField.name = "reason";
        reasonField.typeName = "ReasoningResult!";
        reasonField.description = "Perform AGI reasoning on a question";

        GraphQLArgument reasonInputArg;
        reasonInputArg.name = "input";
        reasonInputArg.typeName = "ReasoningRequest!";
        reasonInputArg.required = true;
        reasonField.arguments.push_back(reasonInputArg);

        queryIt->second.fields.push_back(reasonField);
    }
}

void SchemaRegistry::addLanceDBTypes() {
    // Add Vector scalar
    GraphQLTypeDefinition vectorScalar;
    vectorScalar.name = "Vector";
    vectorScalar.kind = GraphQLTypeKind::Scalar;
    vectorScalar.description = "A vector embedding (list of floats)";
    types_[vectorScalar.name] = vectorScalar;

    // Add VectorSearchResult
    GraphQLTypeDefinition vectorSearchResult;
    vectorSearchResult.name = "SemanticSearchResult";
    vectorSearchResult.kind = GraphQLTypeKind::Object;
    vectorSearchResult.description = "Result from semantic/vector search";

    vectorSearchResult.fields = {
        {"id", "ID!", "Entity ID", {}, false, "", false, false, ""},
        {"score", "Float!", "Similarity score", {}, false, "", false, false, ""},
        {"entityType", "String!", "Type of the entity", {}, false, "", false, false, ""},
        {"content", "String", "Text content", {}, false, "", false, false, ""},
        {"metadata", "JSON", "Additional metadata", {}, false, "", false, false, ""}
    };

    types_[vectorSearchResult.name] = vectorSearchResult;

    // Add VectorStoreInput
    GraphQLTypeDefinition vectorStoreInput;
    vectorStoreInput.name = "VectorStoreInput";
    vectorStoreInput.kind = GraphQLTypeKind::InputObject;
    vectorStoreInput.description = "Input for storing vectors";

    vectorStoreInput.fields = {
        {"entityId", "ID!", "Entity ID to associate with vector", {}, false, "", false, false, ""},
        {"entityType", "String!", "Type of the entity", {}, false, "", false, false, ""},
        {"text", "String", "Text to embed (if not providing vector)", {}, false, "", false, false, ""},
        {"vector", "[Float!]", "Pre-computed vector", {}, false, "", false, false, ""},
        {"metadata", "JSON", "Additional metadata", {}, false, "", false, false, ""}
    };

    types_[vectorStoreInput.name] = vectorStoreInput;
}

void SchemaRegistry::addLadybugTypes() {
    // Add QueryAnalysis
    GraphQLTypeDefinition queryAnalysis;
    queryAnalysis.name = "QueryAnalysis";
    queryAnalysis.kind = GraphQLTypeKind::Object;
    queryAnalysis.description = "Analysis of a query by Ladybug";

    queryAnalysis.fields = {
        {"valid", "Boolean!", "Whether the query is valid", {}, false, "", false, false, ""},
        {"complexity", "Float!", "Query complexity score", {}, false, "", false, false, ""},
        {"estimatedCost", "Float!", "Estimated execution cost", {}, false, "", false, false, ""},
        {"warnings", "[String!]!", "Warning messages", {}, false, "", false, false, ""},
        {"suggestions", "[String!]!", "Optimization suggestions", {}, false, "", false, false, ""},
        {"queryPlan", "String", "Query plan visualization", {}, false, "", false, false, ""}
    };

    types_[queryAnalysis.name] = queryAnalysis;

    // Add QueryMetrics
    GraphQLTypeDefinition queryMetrics;
    queryMetrics.name = "QueryMetrics";
    queryMetrics.kind = GraphQLTypeKind::Object;
    queryMetrics.description = "Performance metrics for a query";

    queryMetrics.fields = {
        {"queryId", "ID!", "Query identifier", {}, false, "", false, false, ""},
        {"parseTimeMs", "Float!", "Parse time in milliseconds", {}, false, "", false, false, ""},
        {"planTimeMs", "Float!", "Planning time in milliseconds", {}, false, "", false, false, ""},
        {"executionTimeMs", "Float!", "Execution time in milliseconds", {}, false, "", false, false, ""},
        {"totalTimeMs", "Float!", "Total time in milliseconds", {}, false, "", false, false, ""},
        {"rowsScanned", "Int!", "Number of rows scanned", {}, false, "", false, false, ""},
        {"rowsReturned", "Int!", "Number of rows returned", {}, false, "", false, false, ""},
        {"cacheHits", "Int!", "Number of cache hits", {}, false, "", false, false, ""},
        {"cacheMisses", "Int!", "Number of cache misses", {}, false, "", false, false, ""}
    };

    types_[queryMetrics.name] = queryMetrics;

    // Add to Query type
    auto queryIt = types_.find(QUERY_TYPE);
    if (queryIt != types_.end()) {
        GraphQLFieldDefinition analyzeField;
        analyzeField.name = "analyzeQuery";
        analyzeField.typeName = "QueryAnalysis!";
        analyzeField.description = "Analyze a query using Ladybug";

        GraphQLArgument queryArg;
        queryArg.name = "query";
        queryArg.typeName = "String!";
        queryArg.required = true;
        analyzeField.arguments.push_back(queryArg);

        queryIt->second.fields.push_back(analyzeField);
    }
}

void SchemaRegistry::addBuiltInScalars() {
    // ID
    GraphQLTypeDefinition idType;
    idType.name = "ID";
    idType.kind = GraphQLTypeKind::Scalar;
    idType.description = "Unique identifier";
    types_[idType.name] = idType;

    // String
    GraphQLTypeDefinition stringType;
    stringType.name = "String";
    stringType.kind = GraphQLTypeKind::Scalar;
    stringType.description = "UTF-8 string";
    types_[stringType.name] = stringType;

    // Int
    GraphQLTypeDefinition intType;
    intType.name = "Int";
    intType.kind = GraphQLTypeKind::Scalar;
    intType.description = "Signed 32-bit integer";
    types_[intType.name] = intType;

    // Float
    GraphQLTypeDefinition floatType;
    floatType.name = "Float";
    floatType.kind = GraphQLTypeKind::Scalar;
    floatType.description = "Double-precision floating point";
    types_[floatType.name] = floatType;

    // Boolean
    GraphQLTypeDefinition boolType;
    boolType.name = "Boolean";
    boolType.kind = GraphQLTypeKind::Scalar;
    boolType.description = "Boolean value";
    types_[boolType.name] = boolType;

    // DateTime
    GraphQLTypeDefinition dateTimeType;
    dateTimeType.name = "DateTime";
    dateTimeType.kind = GraphQLTypeKind::Scalar;
    dateTimeType.description = "ISO 8601 date/time";
    types_[dateTimeType.name] = dateTimeType;

    // JSON
    GraphQLTypeDefinition jsonType;
    jsonType.name = "JSON";
    jsonType.kind = GraphQLTypeKind::Scalar;
    jsonType.description = "Arbitrary JSON value";
    types_[jsonType.name] = jsonType;
}

void SchemaRegistry::addBuiltInDirectives() {
    // @deprecated
    GraphQLDirective deprecated;
    deprecated.name = "deprecated";
    deprecated.description = "Marks an element as deprecated";
    deprecated.locations = {"FIELD_DEFINITION", "ENUM_VALUE"};

    GraphQLArgument reasonArg;
    reasonArg.name = "reason";
    reasonArg.typeName = "String";
    reasonArg.defaultValue = "\"No longer supported\"";
    deprecated.arguments.push_back(reasonArg);

    directives_[deprecated.name] = deprecated;

    // @include
    GraphQLDirective include;
    include.name = "include";
    include.description = "Include field if condition is true";
    include.locations = {"FIELD", "FRAGMENT_SPREAD", "INLINE_FRAGMENT"};

    GraphQLArgument ifArg;
    ifArg.name = "if";
    ifArg.typeName = "Boolean!";
    ifArg.required = true;
    include.arguments.push_back(ifArg);

    directives_[include.name] = include;

    // @skip
    GraphQLDirective skip;
    skip.name = "skip";
    skip.description = "Skip field if condition is true";
    skip.locations = {"FIELD", "FRAGMENT_SPREAD", "INLINE_FRAGMENT"};
    skip.arguments.push_back(ifArg);

    directives_[skip.name] = skip;

    // @semantic - custom directive for semantic search
    GraphQLDirective semantic;
    semantic.name = "semantic";
    semantic.description = "Enable semantic search on this field";
    semantic.locations = {"FIELD_DEFINITION"};

    GraphQLArgument modelArg;
    modelArg.name = "model";
    modelArg.typeName = "String";
    modelArg.defaultValue = "\"text-embedding-3-large\"";
    semantic.arguments.push_back(modelArg);

    directives_[semantic.name] = semantic;
}

void SchemaRegistry::addRelayTypes() {
    // Node interface
    GraphQLTypeDefinition nodeInterface;
    nodeInterface.name = "Node";
    nodeInterface.kind = GraphQLTypeKind::Interface;
    nodeInterface.description = "Relay Node interface";

    GraphQLFieldDefinition idField;
    idField.name = "id";
    idField.typeName = "ID!";
    idField.description = "Globally unique identifier";
    nodeInterface.fields.push_back(idField);

    types_[nodeInterface.name] = nodeInterface;

    // PageInfo
    GraphQLTypeDefinition pageInfo;
    pageInfo.name = "PageInfo";
    pageInfo.kind = GraphQLTypeKind::Object;
    pageInfo.description = "Relay PageInfo";

    pageInfo.fields = {
        {"hasNextPage", "Boolean!", "Whether there are more pages", {}, false, "", false, false, ""},
        {"hasPreviousPage", "Boolean!", "Whether there are previous pages", {}, false, "", false, false, ""},
        {"startCursor", "String", "Cursor for first edge", {}, false, "", false, false, ""},
        {"endCursor", "String", "Cursor for last edge", {}, false, "", false, false, ""}
    };

    types_[pageInfo.name] = pageInfo;
}

std::string SchemaRegistry::kuzuTypeToGraphQL(const std::string& kuzuType) const {
    // Map Kuzu types to GraphQL types
    if (kuzuType.find("INT") != std::string::npos ||
        kuzuType.find("SERIAL") != std::string::npos) {
        return "Int";
    }
    if (kuzuType.find("FLOAT") != std::string::npos ||
        kuzuType.find("DOUBLE") != std::string::npos ||
        kuzuType.find("DECIMAL") != std::string::npos) {
        return "Float";
    }
    if (kuzuType.find("STRING") != std::string::npos ||
        kuzuType.find("VARCHAR") != std::string::npos ||
        kuzuType.find("BLOB") != std::string::npos ||
        kuzuType.find("UUID") != std::string::npos) {
        return "String";
    }
    if (kuzuType.find("BOOL") != std::string::npos) {
        return "Boolean";
    }
    if (kuzuType.find("DATE") != std::string::npos ||
        kuzuType.find("TIME") != std::string::npos ||
        kuzuType.find("INTERVAL") != std::string::npos) {
        return "DateTime";
    }
    if (kuzuType.find("LIST") != std::string::npos ||
        kuzuType.find("[]") != std::string::npos) {
        // Extract inner type
        size_t start = kuzuType.find('[');
        size_t end = kuzuType.find(']');
        if (start != std::string::npos && end != std::string::npos) {
            std::string innerType = kuzuType.substr(start + 1, end - start - 1);
            return "[" + kuzuTypeToGraphQL(innerType) + "]";
        }
        return "[String]";
    }
    if (kuzuType.find("STRUCT") != std::string::npos ||
        kuzuType.find("MAP") != std::string::npos) {
        return "JSON";
    }

    return "String";  // Default fallback
}

// Helper function
std::string SchemaRegistry::toLowerCamelCase(const std::string& str) {
    if (str.empty()) return str;

    std::string result = str;
    result[0] = std::tolower(result[0]);

    // Handle underscores
    size_t writePos = 0;
    bool capitalizeNext = false;
    for (size_t i = 0; i < result.size(); ++i) {
        if (result[i] == '_') {
            capitalizeNext = true;
        } else {
            if (capitalizeNext) {
                result[writePos++] = std::toupper(result[i]);
                capitalizeNext = false;
            } else {
                result[writePos++] = result[i];
            }
        }
    }
    result.resize(writePos);

    return result;
}

} // namespace graphql_agi
} // namespace kuzu
