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

/// Generate out-of-distribution test dataset.
/// These examples contain inputs the model has NEVER seen during training:
/// - Calc: numbers 50-99 (training uses 1-49) and novel phrasings
/// - Weather: unseen cities
/// - Time: unseen timezones
/// - Search: unseen topics
/// - Direct: unseen phrasings
pub fn generate_ood_dataset() -> Vec<ToolExample> {
    let mut ood = Vec::new();

    // === OOD Calc: numbers 50-99 (training range is 1-49) ===
    let ood_calc_examples = [
        ("what is 67 plus 83", "calc(67,83,add)", "the answer is 150"),
        ("compute 91 minus 54", "calc(91,54,sub)", "the answer is 37"),
        ("72 times 88", "calc(72,88,mul)", "the answer is 6336"),
        ("calculate 55 plus 61", "calc(55,61,add)", "the answer is 116"),
        ("what is 99 minus 50", "calc(99,50,sub)", "the answer is 49"),
        ("compute 63 times 77", "calc(63,77,mul)", "the answer is 4851"),
        ("what is 85 plus 92", "calc(85,92,add)", "the answer is 177"),
        ("calculate 78 minus 53", "calc(78,53,sub)", "the answer is 25"),
        ("what is 64 times 56", "calc(64,56,mul)", "the answer is 3584"),
        ("compute 97 plus 88", "calc(97,88,add)", "the answer is 185"),
        ("what is 71 minus 59", "calc(71,59,sub)", "the answer is 12"),
        ("calculate 82 times 93", "calc(82,93,mul)", "the answer is 7626"),
        ("what is 66 plus 74", "calc(66,74,add)", "the answer is 140"),
        ("compute 90 minus 58", "calc(90,58,sub)", "the answer is 32"),
        ("what is 51 times 69", "calc(51,69,mul)", "the answer is 3519"),
    ];

    // Novel phrasings for calc (never seen in training)
    let ood_calc_novel = [
        ("add 55 and 61", "calc(55,61,add)", "the answer is 116"),
        ("sum of 73 and 89", "calc(73,89,add)", "the answer is 162"),
        ("multiply 66 by 77", "calc(66,77,mul)", "the answer is 5082"),
        ("subtract 52 from 95", "calc(95,52,sub)", "the answer is 43"),
        ("find 87 plus 68", "calc(87,68,add)", "the answer is 155"),
    ];

    for &(query, call, reply) in ood_calc_examples.iter().chain(ood_calc_novel.iter()) {
        ood.push(ToolExample {
            input: format!(
                "[user] {} [end] [call] {} [end] [result] {} [end] [reply] {} [end]",
                query, call,
                reply.split("is ").last().unwrap_or("0"),
                reply
            ),
            prompt: format!("[user] {} [end] ", query),
            expected_call: Some(call.to_string()),
            expected_reply: reply.to_string(),
        });
    }

    // === OOD Weather: unseen cities ===
    let ood_weather = [
        ("weather in mumbai", "weather(mumbai)", "the weather in mumbai is 32 degrees and humid"),
        ("what is the weather in toronto", "weather(toronto)", "the weather in toronto is 5 degrees and snowy"),
        ("how is the weather in seoul", "weather(seoul)", "the weather in seoul is 18 degrees and cloudy"),
        ("weather in lagos", "weather(lagos)", "the weather in lagos is 30 degrees and sunny"),
        ("weather in lima", "weather(lima)", "the weather in lima is 22 degrees and clear"),
        ("what is the weather in amsterdam", "weather(amsterdam)", "the weather in amsterdam is 12 degrees and rainy"),
        ("weather in dubai", "weather(dubai)", "the weather in dubai is 38 degrees and sunny"),
        ("how is the weather in stockholm", "weather(stockholm)", "the weather in stockholm is 3 degrees and snowy"),
        ("weather in santiago", "weather(santiago)", "the weather in santiago is 25 degrees and clear"),
        ("what is the weather in buenos aires", "weather(buenos aires)", "the weather in buenos aires is 20 degrees and windy"),
        ("weather in bangkok", "weather(bangkok)", "the weather in bangkok is 33 degrees and humid"),
        ("how is the weather in nairobi", "weather(nairobi)", "the weather in nairobi is 24 degrees and sunny"),
        ("weather in oslo", "weather(oslo)", "the weather in oslo is 1 degrees and snowy"),
        ("what is the weather in cape town", "weather(cape town)", "the weather in cape town is 19 degrees and windy"),
        ("weather in warsaw", "weather(warsaw)", "the weather in warsaw is 8 degrees and cloudy"),
        ("how is the weather in mexico city", "weather(mexico city)", "the weather in mexico city is 21 degrees and clear"),
        ("weather in jakarta", "weather(jakarta)", "the weather in jakarta is 29 degrees and rainy"),
        ("what is the weather in kuala lumpur", "weather(kuala lumpur)", "the weather in kuala lumpur is 31 degrees and humid"),
        ("weather in lisbon", "weather(lisbon)", "the weather in lisbon is 17 degrees and sunny"),
        ("how is the weather in vienna", "weather(vienna)", "the weather in vienna is 10 degrees and cloudy"),
    ];

    for &(query, call, reply) in &ood_weather {
        ood.push(ToolExample {
            input: format!(
                "[user] {} [end] [call] {} [end] [result] {} [end] [reply] {} [end]",
                query, call,
                reply.split("is ").last().unwrap_or("unknown"),
                reply
            ),
            prompt: format!("[user] {} [end] ", query),
            expected_call: Some(call.to_string()),
            expected_reply: reply.to_string(),
        });
    }

    // === OOD Time: unseen timezones ===
    let ood_time = [
        ("what time is it in aest", "time(aest)", "the current time in aest is 14:30"),
        ("time in nzst", "time(nzst)", "the current time in nzst is 16:30"),
        ("current time in brt", "time(brt)", "the current time in brt is 1:30"),
        ("what time is it in kst", "time(kst)", "the current time in kst is 13:30"),
        ("time in mst", "time(mst)", "the current time in mst is 21:30"),
        ("current time in hkt", "time(hkt)", "the current time in hkt is 12:30"),
        ("what time is it in sgt", "time(sgt)", "the current time in sgt is 12:30"),
        ("time in eet", "time(eet)", "the current time in eet is 6:30"),
        ("current time in ast", "time(ast)", "the current time in ast is 0:30"),
        ("what time is it in wet", "time(wet)", "the current time in wet is 4:30"),
        ("time in ict", "time(ict)", "the current time in ict is 11:30"),
        ("current time in wib", "time(wib)", "the current time in wib is 11:30"),
        ("what time is it in cat", "time(cat)", "the current time in cat is 6:30"),
        ("time in cst", "time(cst)", "the current time in cst is 22:30"),
        ("current time in pht", "time(pht)", "the current time in pht is 12:30"),
    ];

    for &(query, call, reply) in &ood_time {
        ood.push(ToolExample {
            input: format!(
                "[user] {} [end] [call] {} [end] [result] {} [end] [reply] {} [end]",
                query, call,
                reply.split("is ").last().unwrap_or("0:00"),
                reply
            ),
            prompt: format!("[user] {} [end] ", query),
            expected_call: Some(call.to_string()),
            expected_reply: reply.to_string(),
        });
    }

    // === OOD Search: unseen topics ===
    let ood_search = [
        ("who discovered penicillin", "search(penicillin discoverer)", "alexander fleming discovered penicillin"),
        ("what is the deepest ocean trench", "search(deepest ocean trench)", "the mariana trench is the deepest ocean trench"),
        ("when was the eiffel tower built", "search(eiffel tower built)", "the eiffel tower was built in 1889"),
        ("who wrote pride and prejudice", "search(pride prejudice author)", "jane austen wrote pride and prejudice"),
        ("what is the smallest country", "search(smallest country)", "vatican city is the smallest country"),
        ("who discovered gravity", "search(gravity discoverer)", "isaac newton discovered gravity"),
        ("what is the longest river", "search(longest river)", "the nile is the longest river"),
        ("when was the great wall built", "search(great wall built)", "the great wall was built over many centuries starting in 7th century bc"),
        ("who invented the light bulb", "search(light bulb inventor)", "thomas edison invented the practical light bulb"),
        ("what is the hottest planet", "search(hottest planet)", "venus is the hottest planet"),
        ("who composed the four seasons", "search(four seasons composer)", "antonio vivaldi composed the four seasons"),
        ("what is the hardest mineral", "search(hardest mineral)", "diamond is the hardest mineral"),
        ("when did the titanic sink", "search(titanic sinking)", "the titanic sank in 1912"),
        ("who founded microsoft", "search(microsoft founder)", "bill gates and paul allen founded microsoft"),
        ("what is the tallest building", "search(tallest building)", "the burj khalifa is the tallest building"),
    ];

    for &(query, call, reply) in &ood_search {
        ood.push(ToolExample {
            input: format!(
                "[user] {} [end] [call] {} [end] [result] {} [end] [reply] {} [end]",
                query, call,
                reply.split("is ").last().unwrap_or("unknown"),
                reply
            ),
            prompt: format!("[user] {} [end] ", query),
            expected_call: Some(call.to_string()),
            expected_reply: reply.to_string(),
        });
    }

    // === OOD Direct: unseen phrasings ===
    let ood_direct = [
        ("hey whats up", "not much how can i help you"),
        ("i appreciate it", "you are welcome happy to help"),
        ("see you later", "goodbye take care"),
        ("good morning", "good morning how can i assist you today"),
        ("nice to meet you", "nice to meet you too how can i help"),
        ("thats awesome", "glad you think so"),
        ("i need some help", "of course what do you need"),
        ("what do you think", "i think i can help you with that"),
        ("can you assist me", "absolutely what would you like to know"),
        ("cheers", "cheers is there anything else you need"),
    ];

    for &(query, reply) in &ood_direct {
        ood.push(ToolExample {
            input: format!("[user] {} [end] [reply] {} [end]", query, reply),
            prompt: format!("[user] {} [end] ", query),
            expected_call: None,
            expected_reply: reply.to_string(),
        });
    }

    ood
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
