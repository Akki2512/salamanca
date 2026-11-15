using UnrealBuildTool;

public class RemoteHandsLink : ModuleRules
{
	public RemoteHandsLink(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine" });
		PrivateDependencyModuleNames.AddRange(new string[] { "Sockets", "Networking", "Json", "HTTPServer", "RenderCore", "RHI" });
		if (Target.bBuildEditor)
		{
			PrivateDependencyModuleNames.Add("UnrealEd");      // to repaint the level viewports when a packet arrives
		}
	}
}
