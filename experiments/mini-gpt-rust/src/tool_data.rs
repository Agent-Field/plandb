use rand::Rng;

/// Tool calling format for mini GPT.
/// Uses ASCII-safe delimiters that fit in our char-level vocab.
///
/// Format:
///   [user] what is 3 plus 5 [end]
///   [call] calc(3,5,add) [end]
///   [result] 8 [end]
///   [reply] the answer is 8 [end]
///
/// Direct answer (no tool needed):
///   [user] hello there [end]
///   [reply] hello how can i help [end]

/// A single training example
#[derive(Clone, Debug)]
pub struct ToolExample {
    pub input: String,   // full sequence for training
    pub prompt: String,  // just the user query part (for eval)
    pub expected_call: Option<String>, // expected tool call (None = direct answer)
    pub expected_reply: String,
}

/// Tool types
#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
pub enum Tool {
    Calc,
    Search,
    Weather,
    Time,
    None, // direct answer
}

/// Evaluation result for a single example
#[derive(Clone, Debug, Default)]
pub struct EvalResult {
    pub format_correct: bool,   // valid [call]...[end] or [reply]...[end] syntax
    pub tool_correct: bool,     // right tool selected (or correctly no tool)
    pub params_correct: bool,   // parameters match
    pub reply_quality: f32,     // 0-1 score for reply (simple heuristic)
}

/// Generate the full synthetic dataset
pub fn generate_dataset(rng: &mut impl Rng) -> (Vec<ToolExample>, Vec<ToolExample>) {
    let mut train = Vec::new();
    let val;

    // Calculator examples
    let ops = [("add", "+", "plus"), ("sub", "-", "minus"), ("mul", "*", "times")];
    for _ in 0..200 {
        let a = rng.gen_range(1..50);
        let b = rng.gen_range(1..50);
        let (op_name, _op_sym, op_word) = ops[rng.gen_range(0..ops.len())];
        let result = match op_name {
            "add" => a + b,
            "sub" => a - b,
            "mul" => a * b,
            _ => 0,
        };

        let queries = [
            format!("what is {} {} {}", a, op_word, b),
            format!("compute {} {} {}", a, op_word, b),
            format!("calculate {} {} {}", a, op_word, b),
            format!("{} {} {}", a, op_word, b),
        ];
        let query = &queries[rng.gen_range(0..queries.len())];
        let call = format!("calc({},{},{})", a, b, op_name);
        let replies = [
            format!("the answer is {}", result),
            format!("{} {} {} is {}", a, op_word, b, result),
            format!("that equals {}", result),
        ];
        let reply = &replies[rng.gen_range(0..replies.len())];

        train.push(ToolExample {
            input: format!(
                "[user] {} [end] [call] {} [end] [result] {} [end] [reply] {} [end]",
                query, call, result, reply
            ),
            prompt: format!("[user] {} [end] ", query),
            expected_call: Some(call.clone()),
            expected_reply: reply.clone(),
        });
    }

    // Search examples
    let search_topics = [
        ("who wrote hamlet", "search(hamlet author)", "hamlet was written by william shakespeare"),
        ("what is the capital of france", "search(capital france)", "the capital of france is paris"),
        ("how tall is mount everest", "search(height everest)", "mount everest is 8849 meters tall"),
        ("when was the moon landing", "search(moon landing date)", "the moon landing was in 1969"),
        ("who invented the telephone", "search(telephone inventor)", "alexander graham bell invented the telephone"),
        ("what is the speed of light", "search(speed of light)", "the speed of light is 299792458 meters per second"),
        ("how many planets are there", "search(number of planets)", "there are 8 planets in our solar system"),
        ("what is the largest ocean", "search(largest ocean)", "the pacific ocean is the largest ocean"),
        ("who painted the mona lisa", "search(mona lisa painter)", "leonardo da vinci painted the mona lisa"),
        ("what year did world war 2 end", "search(ww2 end year)", "world war 2 ended in 1945"),
    ];
    for _ in 0..150 {
        let (query, call, answer) = search_topics[rng.gen_range(0..search_topics.len())];
        let prefixes = ["", "please tell me ", "can you find out ", "i want to know "];
        let prefix = prefixes[rng.gen_range(0..prefixes.len())];
        let full_query = format!("{}{}", prefix, query);
        let result_text = answer.split(" is ").last().unwrap_or(answer);

        train.push(ToolExample {
            input: format!(
                "[user] {} [end] [call] {} [end] [result] {} [end] [reply] {} [end]",
                full_query, call, result_text, answer
            ),
            prompt: format!("[user] {} [end] ", full_query),
            expected_call: Some(call.to_string()),
            expected_reply: answer.to_string(),
        });
    }

    // Weather examples
    let cities = [
        "london", "paris", "tokyo", "new york", "sydney",
        "berlin", "rome", "moscow", "beijing", "cairo",
    ];
    let conditions = ["sunny", "cloudy", "rainy", "snowy", "windy", "clear"];
    for _ in 0..150 {
        let city = cities[rng.gen_range(0..cities.len())];
        let temp = rng.gen_range(0..35);
        let cond = conditions[rng.gen_range(0..conditions.len())];
        let queries = [
            format!("what is the weather in {}", city),
            format!("how is the weather in {}", city),
            format!("weather in {}", city),
            format!("is it {} in {}", cond, city),
        ];
        let query = &queries[rng.gen_range(0..queries.len())];
        let call = format!("weather({})", city);
        let result = format!("{} degrees {}", temp, cond);
        let reply = format!("the weather in {} is {} degrees and {}", city, temp, cond);

        train.push(ToolExample {
            input: format!(
                "[user] {} [end] [call] {} [end] [result] {} [end] [reply] {} [end]",
                query, call, result, reply
            ),
            prompt: format!("[user] {} [end] ", query),
            expected_call: Some(call),
            expected_reply: reply,
        });
    }

    // Time examples
    let zones = ["utc", "est", "pst", "gmt", "cet", "jst", "ist"];
    for _ in 0..100 {
        let zone = zones[rng.gen_range(0..zones.len())];
        let hour = rng.gen_range(0..24);
        let min = rng.gen_range(0..60);
        let queries = [
            format!("what time is it in {}", zone),
            format!("current time in {}", zone),
            format!("time in {}", zone),
        ];
        let query = &queries[rng.gen_range(0..queries.len())];
        let call = format!("time({})", zone);
        let result = format!("{}:{:02}", hour, min);
        let reply = format!("the current time in {} is {}:{:02}", zone, hour, min);

        train.push(ToolExample {
            input: format!(
                "[user] {} [end] [call] {} [end] [result] {} [end] [reply] {} [end]",
                query, call, result, reply
            ),
            prompt: format!("[user] {} [end] ", query),
            expected_call: Some(call),
            expected_reply: reply,
        });
    }

    // Direct answer examples (NO tool call needed)
    let direct = [
        ("hello", "hello how can i help you"),
        ("hi there", "hi nice to meet you"),
        ("thank you", "you are welcome"),
        ("goodbye", "goodbye have a nice day"),
        ("how are you", "i am doing well thank you"),
        ("what can you do", "i can search for information calculate numbers check weather and tell time"),
        ("who are you", "i am a helpful assistant"),
        ("help me", "sure what do you need help with"),
        ("thanks for that", "glad i could help"),
        ("that is great", "happy to hear that"),
        ("ok", "is there anything else i can help with"),
        ("yes please", "sure go ahead and ask"),
        ("no thanks", "alright let me know if you need anything"),
        ("tell me a joke", "why did the chicken cross the road to get to the other side"),
        ("say something nice", "you are doing a great job"),
    ];
    for _ in 0..200 {
        let (query, reply) = direct[rng.gen_range(0..direct.len())];
        train.push(ToolExample {
            input: format!("[user] {} [end] [reply] {} [end]", query, reply),
            prompt: format!("[user] {} [end] ", query),
            expected_call: None,
            expected_reply: reply.to_string(),
        });
    }

    // Shuffle
    shuffle_vec(&mut train, rng);

    // Split: 90% train, 10% val
    let split_point = (train.len() as f32 * 0.9) as usize;
    val = train.split_off(split_point);

    (train, val)
}

fn shuffle_vec<T>(v: &mut Vec<T>, rng: &mut impl Rng) {
    let len = v.len();
    for i in (1..len).rev() {
        let j = rng.gen_range(0..=i);
        v.swap(i, j);
    }
}

/// Build combined vocabulary from base text + tool data
pub fn build_combined_vocab(base_text: &str, examples: &[ToolExample]) -> String {
    let mut all_text = base_text.to_string();
    for ex in examples {
        all_text.push_str(&ex.input);
        all_text.push('\n');
    }
    all_text
}

/// Evaluate model output against expected
pub fn evaluate_output(generated: &str, example: &ToolExample) -> EvalResult {
    let mut result = EvalResult::default();

    if let Some(ref expected_call) = example.expected_call {
        // Should have generated a tool call
        if generated.contains("[call] ") && generated.contains(" [end]") {
            result.format_correct = true;

            // Extract the call
            if let Some(call_start) = generated.find("[call] ") {
                let after_call = &generated[call_start + 7..];
                if let Some(call_end) = after_call.find(" [end]") {
                    let actual_call = &after_call[..call_end];

                    // Check tool name
                    let expected_tool = expected_call.split('(').next().unwrap_or("");
                    let actual_tool = actual_call.split('(').next().unwrap_or("");
                    result.tool_correct = expected_tool == actual_tool;

                    // Check params
                    result.params_correct = actual_call == expected_call.as_str();
                }
            }
        }
    } else {
        // Should have generated a direct reply (no tool call)
        if !generated.contains("[call]") && generated.contains("[reply]") {
            result.format_correct = true;
            result.tool_correct = true; // correctly decided not to call a tool
            result.params_correct = true;
        }
    }

    // Reply quality: simple word overlap
    let gen_lower = generated.to_lowercase();
    let expected_words: Vec<&str> = example.expected_reply.split_whitespace().collect();
    if !expected_words.is_empty() {
        let matches = expected_words.iter()
            .filter(|w| gen_lower.contains(&w.to_lowercase()))
            .count();
        result.reply_quality = matches as f32 / expected_words.len() as f32;
    }

    result
}

/// Aggregate evaluation results
#[derive(Clone, Debug, Default)]
pub struct AggregateMetrics {
    pub format_acc: f32,
    pub tool_acc: f32,
    pub param_acc: f32,
    pub reply_quality: f32,
    pub count: usize,
}

impl AggregateMetrics {
    pub fn from_results(results: &[EvalResult]) -> Self {
        let n = results.len() as f32;
        if results.is_empty() {
            return Self::default();
        }
        Self {
            format_acc: results.iter().filter(|r| r.format_correct).count() as f32 / n,
            tool_acc: results.iter().filter(|r| r.tool_correct).count() as f32 / n,
            param_acc: results.iter().filter(|r| r.params_correct).count() as f32 / n,
            reply_quality: results.iter().map(|r| r.reply_quality).sum::<f32>() / n,
            count: results.len(),
        }
    }

    pub fn print(&self, label: &str) {
        println!("  {} ({} examples):", label, self.count);
        println!("    Format accuracy:  {:.1}%", self.format_acc * 100.0);
        println!("    Tool accuracy:    {:.1}%", self.tool_acc * 100.0);
        println!("    Param accuracy:   {:.1}%", self.param_acc * 100.0);
        println!("    Reply quality:    {:.1}%", self.reply_quality * 100.0);
    }
}
