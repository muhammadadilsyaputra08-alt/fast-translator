package com.fasttranslator

/** Execution backend for on-device inference. Ordinal must match engine::Backend in engine.h. */
enum class Backend {
    QNN,
    VULKAN,
    EXECUTORCH
}

/**
 * Grammar constraint for the sampler. [JSON] guarantees syntactically valid
 * JSON output; [Custom] accepts a raw GBNF grammar string.
 */
sealed class Grammar {
    object None : Grammar()
    object JSON : Grammar()
    data class Custom(val gbnf: String) : Grammar()

    internal fun toGbnfOrNull(): String? = when (this) {
        is None -> null
        is JSON -> DEFAULT_JSON_GBNF
        is Custom -> gbnf
    }

    private companion object {
        // Minimal JSON grammar sufficient to constrain output to valid JSON.
        const val DEFAULT_JSON_GBNF = """
            root   ::= object
            object ::= "{" pair ("," pair)* "}" | "{" "}"
            pair   ::= string ":" value
            value  ::= object | array | string | number | "true" | "false" | "null"
            array  ::= "[" value ("," value)* "]" | "[" "]"
            string ::= "\"" [^"]* "\""
            number ::= "-"? [0-9]+ ("." [0-9]+)?
        """
    }
}

data class KVCacheConfig(
    val enableMiniCache: Boolean = true,
    val enableRocketKV: Boolean = false,
    val compressEvery: Int = 16
)
