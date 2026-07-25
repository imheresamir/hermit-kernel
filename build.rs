use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

use anyhow::{Context, Result, anyhow};
use llvm_tools::LlvmTools;

fn main() -> Result<()> {
	built::write_built_file().unwrap();

	// Assembly embedded via `global_asm!(include_str!("..."))` lives inside the
	// including module's codegen unit. Incremental compilation does NOT reliably
	// invalidate that unit when ONLY the .s file changes (the .rs file is
	// unchanged), so edits to these files can silently link STALE assembly.
	// Declaring them here makes cargo rebuild the crate whenever they change.
	match env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
		"aarch64" => {
			// global_asm!(include_str!("start.s")) in src/arch/aarch64/kernel/mod.rs
			println!("cargo:rerun-if-changed=src/arch/aarch64/kernel/start.s");
		}
		"riscv64" => {
			// global_asm!(include_str!("switch.s")) in src/arch/riscv64/kernel/switch.rs
			println!("cargo:rerun-if-changed=src/arch/riscv64/kernel/switch.s");
		}
		_ => {}
	}

	if env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "x86_64"
		&& env::var_os("CARGO_FEATURE_SMP").is_some()
	{
		assemble_x86_64_smp_boot()?;
	}

	Ok(())
}

fn assemble_x86_64_smp_boot() -> Result<()> {
	let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());

	let boot_s = Path::new("src/arch/x86_64/kernel/boot.s");
	let boot_ll = out_dir.join("boot.ll");
	let boot_bc = out_dir.join("boot.bc");
	let boot_bin = out_dir.join("boot.bin");

	let llvm_as = binutil("llvm-as")?;
	let rust_lld = binutil("rust-lld")?;

	let assembly = fs::read_to_string(boot_s)?;

	let mut llvm_file = File::create(&boot_ll)?;
	writeln!(
		&mut llvm_file,
		r#"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-none-elf"

module asm "
{assembly}
"
"#
	)?;
	llvm_file.flush()?;
	drop(llvm_file);

	let status = Command::new(&llvm_as)
		.arg("-o")
		.arg(&boot_bc)
		.arg(boot_ll)
		.status()
		.with_context(|| format!("Failed to run llvm-as from {}", llvm_as.display()))?;
	assert!(status.success());

	let status = Command::new(&rust_lld)
		.arg("-flavor")
		.arg("gnu")
		.arg("--image-base=0x8000")
		.arg("--section-start=.text=0x8000")
		.arg("--oformat=binary")
		.arg("-o")
		.arg(&boot_bin)
		.arg(&boot_bc)
		.status()
		.with_context(|| format!("Failed to run rust-lld from {}", rust_lld.display()))?;
	assert!(status.success());

	println!("cargo:rerun-if-changed={}", boot_s.display());
	Ok(())
}

fn binutil(name: &str) -> Result<PathBuf> {
	let exe = format!("{name}{}", env::consts::EXE_SUFFIX);

	let path = LlvmTools::new()
		.map_err(|err| match err {
			llvm_tools::Error::NotFound => anyhow!(
				"Could not find llvm-tools component\n\
				\n\
				Maybe the rustup component `llvm-tools` is missing? Install it through: `rustup component add llvm-tools`"
			),
			err => anyhow!("{err:?}"),
		})?
		.tool(&exe)
		.ok_or_else(|| anyhow!("could not find {exe}"))?;

	Ok(path)
}
