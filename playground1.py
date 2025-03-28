# Meh just to test stuff. Nothing important here

from multihopkg.utils.wandb import fix_namespace_duplicates

print("Meepp")
names = ["this.is.one","this.is.other","this.is.it", "that", "that.is","is.it", "this"]
print(fix_namespace_duplicates(names))
